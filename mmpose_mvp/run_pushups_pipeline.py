"""
mmpose_mvp/run_pushups_pipeline.py

Единая точка входа для пайплайна pushups.

Вход:
- user_video_src (обязательно): путь к видео пользователя
- reference_video_src (опционально): путь к видео эталона
  (если передан — сохраняется как reference/reference.mp4 и пересобирается эталон)
- gemini_api_key (опционально): если None — берём из env GEMINI_API_KEY

Выход (артефакты на диске):
- pushups/user/comparison_profiles.png
- pushups/user/_coach/coach_prompt.txt
- pushups/user/_coach/coach_facts.json
- pushups/user/_coach/coach_answer.txt
- pushups/user/last_metrics.json (+ npz)
- pushups/reference/ideal_reference_profile.json (+ npz), если нужно пересобрать эталон
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from mmpose_mvp.gemini_coach_client import (
    GeminiCallConfig,
    GeminiCoachClient,
    save_text,
)
from mmpose_mvp.llm_coach import build_llm_prompt_bundle, load_last_metrics
from mmpose_mvp.plots import save_comparison_profiles_canvas
from mmpose_mvp.reference_builder import build_and_save_reference
from mmpose_mvp.storage import (
    ensure_reference_exists,
    get_exercise_paths,
    save_video_bytes,
)
from mmpose_mvp.user_analyzer import (
    UserAnalysisConfig,
    analyze_user_video_and_compare,
)


@dataclass(frozen=True)
class PushupsPipelineResult:
    comparison_png: Path
    last_metrics_json: Path
    coach_prompt_txt: Path
    coach_facts_json: Path
    coach_answer_txt: Path
    reference_json: Path


def _get_fps(video_path: Path, fallback: float = 25.0) -> float:
    """Читает FPS через OpenCV (с fallback)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or fallback
    cap.release()
    return float(fps)


def run_pushups_pipeline(
    *,
    base_dir: str | Path = "mmpose_mvp/data_mvp",
    user_video_src: str | Path,
    reference_video_src: str | Path | None = None,
    gemini_api_key: str | None = None,
    device: str = "cpu",
    # LLM / prompt настройки
    top_k: int = 6,
    min_recommendations: int = 4,
    max_recommendations: int = 6,
    # Gemini настройки
    gemini_cfg: GeminiCallConfig = GeminiCallConfig(
        max_output_tokens=8000, temperature=0.4
    ),
) -> PushupsPipelineResult:
    """
    ЕДИНАЯ точка входа:

    - reference_video_src: если передан -> сохраняем как reference/reference.mp4 и пересобираем эталон
    - если reference_video_src не передан -> используем существующий эталон (если он есть)
    - user_video_src обязателен -> сохраняем как user/last_attempt.mp4
    """
    paths = get_exercise_paths(base_dir, "pushups")

    # --- user video (обязательно) -> user/last_attempt.mp4
    user_video_src = Path(user_video_src)
    if not user_video_src.exists():
        raise FileNotFoundError(f"user_video_src not found: {user_video_src}")
    save_video_bytes(user_video_src.read_bytes(), paths.user_video)

    # --- reference video (опционально) -> reference/reference.mp4
    reference_was_updated = False
    if reference_video_src is not None:
        ref_src = Path(reference_video_src)
        if not ref_src.exists():
            raise FileNotFoundError(
                f"reference_video_src not found: {ref_src}"
            )

        # ВАЖНО: фиксированный путь
        save_video_bytes(ref_src.read_bytes(), paths.reference_video)
        reference_was_updated = True

    # --- ensure / (re)build reference profiles if needed
    if reference_was_updated or (not paths.reference_json.exists()):
        if not paths.reference_video.exists():
            raise FileNotFoundError(
                "reference video missing. Pass reference_video_src or ensure reference/reference.mp4 exists."
            )

        fps_ref = _get_fps(paths.reference_video)
        build_and_save_reference(
            video_path=paths.reference_video,
            out_npz_path=paths.reference_kpts_npz,
            out_json_path=paths.reference_json,
            fps=fps_ref,
            device=device,
            save_debug_vis=False,
            debug_dir=(paths.reference_dir / "_debug"),
        )

    # Если эталон не передавали и json уже был — просто проверим, что он есть
    ensure_reference_exists(paths)

    # --- analyze user vs reference -> last_metrics.json (+ user npz)
    fps_user = _get_fps(paths.user_video)
    cfg = UserAnalysisConfig(device=device, save_debug_vis=False)

    _ = analyze_user_video_and_compare(
        user_video_path=paths.user_video,
        ref_json_path=paths.reference_json,
        fps=fps_user,
        out_user_npz_path=paths.user_kpts_npz,
        out_metrics_json_path=paths.user_metrics_json,
        cfg=cfg,
        debug_dir=(paths.user_dir / "_debug"),
    )

    # --- comparison plot -> comparison_profiles.png (через ТВОЙ plots.py)
    comparison_png = paths.user_dir / "comparison_profiles.png"
    save_comparison_profiles_canvas(
        ref_json_path=paths.reference_json,
        user_metrics_json_path=paths.user_metrics_json,
        out_png=comparison_png,
        band_sigma=1.0,
        dpi=150,
    )

    # --- build prompt + facts
    coach_dir = paths.user_dir / "_coach"
    coach_dir.mkdir(parents=True, exist_ok=True)

    lm = load_last_metrics(paths.user_metrics_json)
    bundle = build_llm_prompt_bundle(
        lm,
        top_k=int(top_k),
        min_recommendations=int(min_recommendations),
        max_recommendations=int(max_recommendations),
    )

    prompt_path = coach_dir / "coach_prompt.txt"
    prompt_path.write_text(bundle.prompt, encoding="utf-8")

    facts_path = coach_dir / "coach_facts.json"
    facts_path.write_text(
        json.dumps(bundle.facts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- Gemini call
    api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key missing. Pass gemini_api_key=... or set env GEMINI_API_KEY"
        )

    client = GeminiCoachClient(api_key=api_key)
    answer_text = client.generate_text(bundle.prompt, cfg=gemini_cfg)

    answer_path = coach_dir / "coach_answer.txt"
    save_text(answer_text, answer_path)

    return PushupsPipelineResult(
        comparison_png=comparison_png,
        last_metrics_json=paths.user_metrics_json,
        coach_prompt_txt=prompt_path,
        coach_facts_json=facts_path,
        coach_answer_txt=answer_path,
        reference_json=paths.reference_json,
    )
