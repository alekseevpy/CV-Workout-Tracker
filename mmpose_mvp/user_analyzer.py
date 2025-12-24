"""
mmpose_mvp/user_analyzer.py

Модуль анализа пользовательского выполнения упражнения и сравнения с эталоном.

Что делает модуль:
1) Прогоняет видео пользователя через MMPose (pose2d) и извлекает скелет (COCO keypoints).
2) Заполняет пропуски (NaN) и нормализует скелет так же, как для эталона:
   - центрирование по центру таза (mid-hip)
   - масштабирование по ширине плеч (shoulder width)
3) Вычисляет временные ряды углов суставов (локти/плечи/бедра/колени/наклон корпуса).
4) Очищает ряды (deglitch + smooth, для trunk_tilt unwrap).
5) Находит повторы на видео пользователя (по максимумам среднего угла локтей).
6) Каждый повтор переводит в "фазу" 0..100% (ресэмпл до PHASE_POINTS).
7) Строит пользовательский профиль как mean/std по всем найденным повторам.
8) Загружает эталон (ideal_reference_profile.json) и сравнивает user mean со reference mean,
   используя reference std как "допуск" (±1σ).
9) Сохраняет артефакты:
   - last_attempt_keypoints.npz (kpts, scores)
   - last_metrics.json (meta + user_profile + агрегированные метрики + метрики по углам)

Важно:
- Здесь пользовательский профиль строится так же, как эталон: mean/std по множеству повторов.
- Сравнение выполняется по фазовым профилям длины phase_points (обычно 101).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from mmpose_mvp.reference_builder import (
    ReferenceBuildConfig,
    clean_and_normalize_skeleton,
    clean_angles,
    compute_angles_series,
    extract_keypoints_from_video,
    find_repetitions_by_elbow,
    resample_repeat,
)


# -------------------------
# Data models
# -------------------------
@dataclass(frozen=True)
class AngleProfile:
    """
    Профиль угла по фазе 0..100%.

    Parameters:
        mean : np.ndarray - Среднее значение угла по фазе (shape: [phase_points]).
        std  : np.ndarray - Стандартное отклонение по фазе (shape: [phase_points]).
    """

    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class UserAnalysisConfig:
    """
    Настройки анализа пользовательского видео.

    Parameters:
        build_cfg : ReferenceBuildConfig
            Настройки извлечения/чистки/фазирования (должны совпадать с эталоном).
        min_reps_required : int
            Минимально требуемое число найденных повторов для "надёжного" профиля.
            Если найдено меньше — анализ всё равно выполняется, но в meta будет предупреждение.
        device : str
            Устройство для MMPoseInferencer ("cpu", "cuda", "mps"...).
        save_debug_vis : bool
            Если True — сохраняются:
              - MMPose vis/preds (через extract_keypoints_from_video)
              - графики debug_elbow_mean.png и debug_elbow_peaks_and_reps.png
    """

    build_cfg: ReferenceBuildConfig = ReferenceBuildConfig()
    min_reps_required: int = 2
    device: str = "cpu"
    save_debug_vis: bool = False


@dataclass(frozen=True)
class CompareMetrics:
    """
    Метрики сравнения одного угла (профиля) пользователя с эталоном.

    Parameters:
        mae : float - Mean Absolute Error между user_mean и ref_mean по фазе.
        max_err : float - Максимальная абсолютная ошибка по фазе.
        pct_outside_1sigma : float - % фазовых точек, где |user_mean - ref_mean| > ref_std.
    """

    mae: float
    max_err: float
    pct_outside_1sigma: float


# -------------------------
# Internal helpers (debug)
# -------------------------
def _ensure_dir(p: str | Path) -> Path:
    """Создаёт директорию (parents=True) и возвращает Path."""
    pp = Path(p)
    pp.mkdir(parents=True, exist_ok=True)
    return pp


def _save_elbow_mean_plot(out_dir: Path, elbow_mean: np.ndarray) -> None:
    """Сохраняет elbow_mean в PNG."""
    out_png = out_dir / "debug_elbow_mean.png"
    plt.figure(figsize=(10, 3))
    plt.plot(elbow_mean)
    plt.title("elbow_mean (clean) — for rep detection")
    plt.xlabel("frame")
    plt.ylabel("deg")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _save_peaks_and_reps_plot(
    out_dir: Path,
    elbow_mean: np.ndarray,
    peaks: np.ndarray,
    reps: List[Tuple[int, int]],
    *,
    prom_deg: float,
    min_peak_dist_frames: int,
    min_rep_len_sec: float,
) -> None:
    """Сохраняет график elbow_mean + пики + границы повторов."""
    out_png = out_dir / "debug_elbow_peaks_and_reps.png"
    plt.figure(figsize=(12, 3))
    plt.plot(elbow_mean, label="elbow_mean")
    if len(peaks) > 0:
        plt.scatter(
            peaks, elbow_mean[peaks], s=25, label=f"peaks={len(peaks)}"
        )
    for s, e in reps:
        plt.axvline(s, linestyle="--", alpha=0.6)
        plt.axvline(e, linestyle="--", alpha=0.6)
    plt.title(
        f"peaks={len(peaks)} | reps={len(reps)} | prom={prom_deg} | "
        f"min_peak_dist={min_peak_dist_frames}f | min_rep_len={min_rep_len_sec}s"
    )
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# -------------------------
# Reference loading
# -------------------------
def load_reference_profiles(
    ref_json_path: str | Path,
) -> Tuple[Dict[str, AngleProfile], dict]:
    """
    Загружает эталонный профиль (ideal_reference_profile.json).

    Parameters:
        ref_json_path : str | Path - Путь до файла ideal_reference_profile.json.
    Returns:
        ref_profiles : dict[str, AngleProfile] - Профили эталона по углам (mean/std).
        meta : dict - meta-блок из эталона (fps, phase_points, angles, ...).
    Raises:
        FileNotFoundError - Если файл не существует.
        KeyError - Если структура файла некорректна.
    """
    ref_json_path = Path(ref_json_path)
    if not ref_json_path.exists():
        raise FileNotFoundError(f"Reference JSON not found: {ref_json_path}")

    with open(ref_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["meta"]
    angles = meta["angles"]

    ref_profiles: Dict[str, AngleProfile] = {}
    for a in angles:
        ref_profiles[a] = AngleProfile(
            mean=np.array(data[a]["mean"], dtype=float),
            std=np.array(data[a]["std"], dtype=float),
        )
    return ref_profiles, meta


# -------------------------
# User profiles building (multi-rep -> mean/std)
# -------------------------
def build_user_phase_profiles(
    video_path: str | Path,
    *,
    fps: float,
    cfg: UserAnalysisConfig = UserAnalysisConfig(),
    out_npz_path: str | Path | None = None,
    debug_dir: str | Path | None = None,
) -> Tuple[Dict[str, AngleProfile], Dict[str, np.ndarray], dict]:
    """
    Строит пользовательский профиль углов (mean/std) по всем найденным повторам.

    Parameters:
        video_path : str | Path - Путь к видео пользователя.
        fps : float - FPS видео пользователя (получаем снаружи через cv2.VideoCapture).
        cfg : UserAnalysisConfig - Настройки анализа.
        out_npz_path : str | Path | None - Если задано — сохранит keypoints в NPZ (kpts, scores).
        debug_dir : str | Path | None - База для debug-артефактов (если cfg.save_debug_vis=True).
    Returns:
        user_profiles : dict[str, AngleProfile] - mean/std по фазе (shape: [phase_points]) для каждого угла.
        per_rep_curves : dict[str, np.ndarray]
            Сырые фазовые профили по повторам:
            shape = (num_reps, phase_points) для каждого угла.
        meta : dict - Метаданные по анализу пользователя.
    Raises:
        RuntimeError - Если MMPose не вернул ни одного кадра.
        ValueError - Если не удалось выделить повторы (reps пуст).
    """
    video_path = Path(video_path)
    build_cfg = cfg.build_cfg

    # debug dirs (optional)
    vis_dir = pred_dir = None
    debug_root: Path | None = None
    if cfg.save_debug_vis:
        debug_root = (
            Path(debug_dir)
            if debug_dir is not None
            else (video_path.parent / "_debug_user")
        )
        debug_root = _ensure_dir(debug_root)
        vis_dir = _ensure_dir(debug_root / "vis_user")
        pred_dir = _ensure_dir(debug_root / "preds_user")

    # 1) keypoints
    kpts, scores = extract_keypoints_from_video(
        video_path,
        device=cfg.device,
        save_vis_dir=vis_dir,
        save_pred_dir=pred_dir,
    )

    # optional save npz
    if out_npz_path is not None:
        out_npz_path = Path(out_npz_path)
        out_npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_npz_path, kpts=kpts, scores=scores)

    # 2) normalize
    k_norm, _scores_clean = clean_and_normalize_skeleton(kpts, scores)

    # 3) angles
    angles = compute_angles_series(k_norm)
    angles_clean = clean_angles(angles, build_cfg)

    elbow_mean = (
        angles_clean["elbow_left"] + angles_clean["elbow_right"]
    ) / 2.0
    if cfg.save_debug_vis and debug_root is not None:
        _save_elbow_mean_plot(debug_root, elbow_mean)

    # 4) repetitions
    reps = find_repetitions_by_elbow(angles_clean, fps=fps, cfg=build_cfg)

    if cfg.save_debug_vis and debug_root is not None:
        min_peak_dist_frames = max(1, int(fps * build_cfg.min_rep_sec))
        peaks, _props = find_peaks(
            elbow_mean,
            prominence=build_cfg.prom_deg,
            distance=min_peak_dist_frames,
        )
        _save_peaks_and_reps_plot(
            debug_root,
            elbow_mean,
            peaks,
            reps,
            prom_deg=build_cfg.prom_deg,
            min_peak_dist_frames=min_peak_dist_frames,
            min_rep_len_sec=build_cfg.min_rep_len_sec,
        )

    if not reps:
        raise ValueError(
            "Не удалось выделить повторы на видео пользователя (reps пуст)."
        )

    # 5) resample each rep to phase_points -> per-rep curves
    per_rep_curves: Dict[str, List[np.ndarray]] = {
        name: [] for name in angles_clean
    }
    for s, e in reps:
        for name, series in angles_clean.items():
            per_rep_curves[name].append(
                resample_repeat(series, s, e, build_cfg.phase_points)
            )

    # 6) mean/std over reps
    per_rep_arrays: Dict[str, np.ndarray] = {}
    user_profiles: Dict[str, AngleProfile] = {}
    for name, curves in per_rep_curves.items():
        if not curves:
            continue
        A = np.stack(curves, axis=0)  # (num_reps, phase_points)
        per_rep_arrays[name] = A
        mean = A.mean(axis=0)
        std = A.std(axis=0, ddof=1) if A.shape[0] > 1 else np.zeros_like(mean)
        user_profiles[name] = AngleProfile(mean=mean, std=std)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_video": str(video_path),
        "fps": float(fps),
        "phase_points": int(build_cfg.phase_points),
        "reps_found": int(len(reps)),
        "reps_used": int(len(reps)),
        "min_reps_required": int(cfg.min_reps_required),
        "warning": (
            None
            if len(reps) >= cfg.min_reps_required
            else f"Low repetitions count: found {len(reps)} < {cfg.min_reps_required}"
        ),
        "angles": sorted(list(user_profiles.keys())),
    }

    return user_profiles, per_rep_arrays, meta


# -------------------------
# Comparison logic
# -------------------------
def compare_profiles_to_reference(
    user_profiles: Dict[str, AngleProfile],
    ref_profiles: Dict[str, AngleProfile],
) -> Tuple[Dict[str, CompareMetrics], Dict[str, float]]:
    """
    Сравнивает user mean (по фазе) с reference mean, используя reference std как "допуск" (±1σ).

    Parameters:
        user_profiles : dict[str, AngleProfile] - Профили пользователя (mean/std по фазе).
        ref_profiles : dict[str, AngleProfile] - Профили эталона (mean/std по фазе).
    Returns:
        metrics_by_angle : dict[str, CompareMetrics] - Метрики по каждому углу.
        summary : dict[str, float] - Агрегированные показатели:
              * overall_mae
              * overall_pct_outside_1sigma
              * angles_compared
    """
    common = [a for a in ref_profiles.keys() if a in user_profiles]
    metrics_by_angle: Dict[str, CompareMetrics] = {}

    for a in common:
        u = user_profiles[a].mean
        r = ref_profiles[a].mean
        rs = ref_profiles[a].std

        err = np.abs(u - r)
        metrics_by_angle[a] = CompareMetrics(
            mae=float(np.mean(err)),
            max_err=float(np.max(err)),
            pct_outside_1sigma=float(np.mean(err > rs) * 100.0),
        )

    if metrics_by_angle:
        overall_mae = float(
            np.mean([m.mae for m in metrics_by_angle.values()])
        )
        overall_out = float(
            np.mean([m.pct_outside_1sigma for m in metrics_by_angle.values()])
        )
    else:
        overall_mae = float("nan")
        overall_out = float("nan")

    summary = {
        "overall_mae": overall_mae,
        "overall_pct_outside_1sigma": overall_out,
        "angles_compared": int(len(metrics_by_angle)),
    }
    return metrics_by_angle, summary


# -------------------------
# Orchestrator: analyze + compare + save json
# -------------------------
def analyze_user_video_and_compare(
    *,
    user_video_path: str | Path,
    ref_json_path: str | Path,
    fps: float,
    out_user_npz_path: str | Path | None = None,
    out_metrics_json_path: str | Path | None = None,
    cfg: UserAnalysisConfig = UserAnalysisConfig(),
    debug_dir: str | Path | None = None,
) -> Dict:
    """
    Полный пайплайн: пользовательское видео -> user mean/std -> сравнение с эталоном -> (опционально) сохранение JSON.

    Parameters:
        user_video_path : str | Path - Путь к видео пользователя.
        ref_json_path : str | Path - Путь к ideal_reference_profile.json.
        fps : float - FPS пользовательского видео.
        out_user_npz_path : str | Path | None - Куда сохранить NPZ с keypoints пользователя (kpts, scores). Если None — не сохраняем.
        out_metrics_json_path : str | Path | None - Куда сохранить JSON с метриками сравнения. Если None — не сохраняем.
        cfg : UserAnalysisConfig - Настройки анализа пользователя.
        debug_dir : str | Path | None - База для debug-вывода (если cfg.save_debug_vis=True).

    Returns:
        dict - JSON-совместимая структура результата.
    """
    ref_profiles, meta_ref = load_reference_profiles(ref_json_path)

    user_profiles, _per_rep, meta_user = build_user_phase_profiles(
        user_video_path,
        fps=fps,
        cfg=cfg,
        out_npz_path=out_user_npz_path,
        debug_dir=debug_dir,
    )

    phase_u = int(meta_user["phase_points"])
    phase_r = int(meta_ref["phase_points"])
    if phase_u != phase_r:
        raise ValueError(
            f"phase_points mismatch: user={phase_u} vs reference={phase_r}. "
            f"Make sure configs match."
        )

    metrics_by_angle, summary = compare_profiles_to_reference(
        user_profiles, ref_profiles
    )

    out: Dict = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "phase_points": phase_u,
        },
        "meta_user": meta_user,
        "meta_reference": meta_ref,
        "summary": summary,
        "metrics_by_angle": {
            k: {
                "mae": float(v.mae),
                "max_err": float(v.max_err),
                "pct_outside_1sigma": float(v.pct_outside_1sigma),
            }
            for k, v in metrics_by_angle.items()
        },
        # ВАЖНО: сохраняем пользовательский профиль (mean/std), как просили "как у эталона"
        "user_profile": {
            a: {
                "mean": user_profiles[a].mean.tolist(),
                "std": user_profiles[a].std.tolist(),
            }
            for a in sorted(user_profiles.keys())
        },
    }

    if out_metrics_json_path is not None:
        out_metrics_json_path = Path(out_metrics_json_path)
        out_metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    return out
