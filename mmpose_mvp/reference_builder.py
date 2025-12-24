"""
mmpose_mvp/reference_builder.py

Модуль построения эталонного профиля упражнения по видео.

Что делает модуль:
1) Прогоняет видео через MMPose (pose2d) и извлекает скелет (COCO keypoints).
2) Заполняет пропуски (NaN) и нормализует скелет:
   - центрирование по центру таза (mid-hip)
   - масштабирование по ширине плеч (shoulder width)
3) Вычисляет временные ряды углов суставов (локти/плечи/бедра/колени/наклон корпуса).
4) Очищает ряды (deglitch + smooth, для trunk_tilt unwrap).
5) Находит повторы на эталонном видео (по максимумам среднего угла локтей).
6) Каждый повтор переводит в "фазу" 0..100% (ресэмпл до PHASE_POINTS).
7) Строит эталон как mean/std по всем найденным повторам.
8) Сохраняет артефакты:
   - reference_keypoints.npz (kpts, scores)
   - ideal_reference_profile.json (meta + mean/std по углам)

Модуль не занимается сравнением с пользователем — только создаёт эталон.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from mmpose.apis import MMPoseInferencer
from scipy.signal import find_peaks, savgol_filter

# COCO-17 индексы
COCO = dict(
    nose=0,
    l_eye=1,
    r_eye=2,
    l_ear=3,
    r_ear=4,
    l_sho=5,
    r_sho=6,
    l_elb=7,
    r_elb=8,
    l_wri=9,
    r_wri=10,
    l_hip=11,
    r_hip=12,
    l_kne=13,
    r_kne=14,
    l_ank=15,
    r_ank=16,
)


@dataclass(frozen=True)
class ReferenceBuildConfig:
    """
    Настройки построения эталона.

    Parameters:
        phase_points : int - Количество точек фазового профиля (обычно 101).
        smooth_win : int - Окно сглаживания (нечётное).
        smooth_poly : int - Степень полинома для Savitzky–Golay (если scipy доступен).
        glitch_thr_deg : float - Порог для выбросов по первой разности угла.
        min_rep_sec : float - Минимальная “дистанция” между верхними точками (сек) при поиске пиков.
        prom_deg : float - Требуемая выраженность пиков (prominence) для find_peaks.
        min_rep_len_sec : float - Минимальная длительность повтора (сек) чтобы считать интервал валидным.
    """

    phase_points: int = 101
    smooth_win: int = 11
    smooth_poly: int = 2
    glitch_thr_deg: float = 35.0
    min_rep_sec: float = 0.25
    prom_deg: float = 5.0
    min_rep_len_sec: float = 0.6


# -------------------------
# 1) Keypoints extraction
# -------------------------
def extract_keypoints_from_video(
    video_path: str | Path,
    *,
    device: str = "cpu",
    save_vis_dir: str | Path | None = None,
    save_pred_dir: str | Path | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Прогоняет видео через MMPoseInferencer и возвращает keypoints + scores.

    Parameters:
        video_path : str | Path - Путь к видео.
        device : str - Устройство для инференса ("cpu", "cuda", "mps" и т.п.).
        save_vis_dir : str | Path | None - Если задано — MMPose сохранит визуализацию скелета поверх видео.
        save_pred_dir : str | Path | None - Если задано — MMPose сохранит покадровые предсказания в json.
    Returns:
        kpts : np.ndarray - Массив формы (T, 17, 2) — координаты COCO-17 (x,y) по кадрам.
        scores : np.ndarray - Массив формы (T, 17) — confidence по кадрам/точкам.
    Notes:
        Если на кадре не найден человек — записываются NaN.
        Если найдено несколько людей — берётся человек с максимальным средним score.
    """
    inferencer = MMPoseInferencer(pose2d="human", device=device)

    gen = inferencer(
        str(video_path),
        vis_out_dir=str(save_vis_dir) if save_vis_dir else None,
        pred_out_dir=str(save_pred_dir) if save_pred_dir else None,
        show=False,
        return_vis=False,
        radius=4,
        thickness=2,
    )

    all_kpts: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []

    for res in gen:
        preds = res.get("predictions", [[]])[0]
        if len(preds) == 0:
            all_kpts.append(np.full((17, 2), np.nan, dtype=np.float32))
            all_scores.append(np.full((17,), np.nan, dtype=np.float32))
        else:
            best = max(
                preds,
                key=lambda p: float(np.mean(p.get("keypoint_scores", [0]))),
            )
            all_kpts.append(np.array(best["keypoints"], dtype=np.float32))
            all_scores.append(
                np.array(best["keypoint_scores"], dtype=np.float32)
            )

    if not all_kpts:
        raise RuntimeError("MMPoseInferencer не вернул ни одного кадра.")

    kpts = np.stack(all_kpts, axis=0)  # (T,17,2)
    scores = np.stack(all_scores, axis=0)  # (T,17)

    return kpts, scores


# -------------------------
# 2) Cleaning + normalization
# -------------------------
def _fill_nans_timewise(arr: np.ndarray) -> np.ndarray:
    """
    Линейно интерполирует NaN по времени (по оси 0).

    Parameters:
        arr : np.ndarray - Входной массив (T,) или (T, D).
    Returns:
        np.ndarray - Заполненный массив той же формы.
    """
    arr = arr.copy()
    t = arr.shape[0]
    if arr.ndim == 1:
        arr = arr[:, None]
    idx = np.arange(t)
    for d in range(arr.shape[1]):
        x = arr[:, d]
        nans = ~np.isfinite(x)
        if nans.any() and (~nans).any():
            x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
        arr[:, d] = x
    return arr.squeeze()


def clean_and_normalize_skeleton(
    kpts: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Чистит NaN (интерполяция по времени) и нормализует скелет.

    Normalization:
      - центр: mid-hip = (l_hip + r_hip)/2 становится (0,0)
      - масштаб: ширина плеч (l_sho-r_sho) приводится к ~1
    Parameters:
        kpts : np.ndarray - (T,17,2) keypoints
        scores : np.ndarray - (T,17) confidence
    Returns:
        k_norm : np.ndarray - (T,17,2) нормализованные координаты
        scores_clean : np.ndarray - (T,17) scores после заполнения NaN
    """
    kpts = kpts.astype(np.float32).copy()
    scores = scores.astype(np.float32).copy()

    for j in range(17):
        kpts[:, j, 0] = _fill_nans_timewise(kpts[:, j, 0])
        kpts[:, j, 1] = _fill_nans_timewise(kpts[:, j, 1])
        scores[:, j] = _fill_nans_timewise(scores[:, j])

    # normalize
    lhip, rhip = COCO["l_hip"], COCO["r_hip"]
    lsho, rsho = COCO["l_sho"], COCO["r_sho"]

    out = kpts.copy()
    mid_hip = (out[:, lhip, :] + out[:, rhip, :]) / 2.0
    out -= mid_hip[:, None, :]

    shoulder_w = np.linalg.norm(out[:, lsho, :] - out[:, rsho, :], axis=1)
    ok = np.isfinite(shoulder_w) & (shoulder_w > 1e-6)
    scale = np.median(shoulder_w[ok]) if ok.any() else 1.0
    shoulder_w = np.where(ok, shoulder_w, scale)
    out /= shoulder_w[:, None, None]

    return out, scores


# -------------------------
# 3) Angles + preprocessing
# -------------------------
def angle_ABC(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, eps: float = 1e-6
) -> float:
    """Возвращает угол ABC в градусах."""
    BA, BC = A - B, C - B
    na, nc = np.linalg.norm(BA) + eps, np.linalg.norm(BC) + eps
    cosv = np.clip(np.dot(BA, BC) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def compute_angles_series(k_norm: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Считает временные ряды углов по нормализованному скелету.

    Returns:
        dict[str, np.ndarray] -
            * Ключи: elbow_left/right, shoulder_left/right,
                     hip_left/right, knee_left/right, trunk_tilt
            * Значения: массивы длины T (по кадрам).
    """
    T = k_norm.shape[0]

    LSH, RSH = COCO["l_sho"], COCO["r_sho"]
    LEL, REL = COCO["l_elb"], COCO["r_elb"]
    LWR, RWR = COCO["l_wri"], COCO["r_wri"]
    LHP, RHP = COCO["l_hip"], COCO["r_hip"]
    LKN, RKN = COCO["l_kne"], COCO["r_kne"]
    LAN, RAN = COCO["l_ank"], COCO["r_ank"]

    mid_sh = (k_norm[:, LSH, :] + k_norm[:, RSH, :]) / 2.0
    mid_hp = (k_norm[:, LHP, :] + k_norm[:, RHP, :]) / 2.0

    A: Dict[str, np.ndarray] = {}
    A["elbow_left"] = np.array(
        [
            angle_ABC(k_norm[t, LSH], k_norm[t, LEL], k_norm[t, LWR])
            for t in range(T)
        ]
    )
    A["elbow_right"] = np.array(
        [
            angle_ABC(k_norm[t, RSH], k_norm[t, REL], k_norm[t, RWR])
            for t in range(T)
        ]
    )
    A["shoulder_left"] = np.array(
        [
            angle_ABC(k_norm[t, LEL], k_norm[t, LSH], k_norm[t, LHP])
            for t in range(T)
        ]
    )
    A["shoulder_right"] = np.array(
        [
            angle_ABC(k_norm[t, REL], k_norm[t, RSH], k_norm[t, RHP])
            for t in range(T)
        ]
    )
    A["hip_left"] = np.array(
        [
            angle_ABC(k_norm[t, LSH], k_norm[t, LHP], k_norm[t, LKN])
            for t in range(T)
        ]
    )
    A["hip_right"] = np.array(
        [
            angle_ABC(k_norm[t, RSH], k_norm[t, RHP], k_norm[t, RKN])
            for t in range(T)
        ]
    )
    A["knee_left"] = np.array(
        [
            angle_ABC(k_norm[t, LHP], k_norm[t, LKN], k_norm[t, LAN])
            for t in range(T)
        ]
    )
    A["knee_right"] = np.array(
        [
            angle_ABC(k_norm[t, RHP], k_norm[t, RKN], k_norm[t, RAN])
            for t in range(T)
        ]
    )

    v = mid_sh - mid_hp
    A["trunk_tilt"] = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    return A


def _interp_nans_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(float).copy()
    n = len(x)
    idx = np.arange(n)
    bad = ~np.isfinite(x)
    if bad.all():
        return np.zeros_like(x)
    if bad.any():
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
        if not np.isfinite(x[0]):
            x[0] = x[np.isfinite(x)][0]
        if not np.isfinite(x[-1]):
            x[-1] = x[np.isfinite(x)][-1]
    return x


def _unwrap_deg(x: np.ndarray, center: float = 0.0) -> np.ndarray:
    rad = np.deg2rad(x - center)
    unwrapped = np.unwrap(rad)
    return np.rad2deg(unwrapped) + center


def _smooth_series(x: np.ndarray, win: int, poly: int) -> np.ndarray:
    x = x.astype(float)
    try:
        w = win if win % 2 == 1 else win + 1
        w = max(5, min(w, len(x) - (1 - len(x) % 2)))
        if w % 2 == 0:
            w -= 1
        return savgol_filter(
            x, window_length=w, polyorder=min(poly, 3), mode="interp"
        )
    except Exception:
        k = max(3, win if win % 2 == 1 else win + 1)
        pad = k // 2
        xx = np.pad(x, (pad, pad), mode="edge")
        ker = np.ones(k) / k
        return np.convolve(xx, ker, mode="valid")


def _deglitch_series(x: np.ndarray, thr_deg: float) -> np.ndarray:
    x = x.astype(float).copy()
    d = np.diff(x, prepend=x[0])
    spikes = np.abs(d) > thr_deg
    if spikes.any():
        x[spikes] = np.nan
        x = _interp_nans_1d(x)
    return x


def clean_angles(
    angles: Dict[str, np.ndarray],
    cfg: ReferenceBuildConfig,
) -> Dict[str, np.ndarray]:
    """
    Чистит ряды углов: deglitch -> unwrap (для trunk_tilt) -> smooth.

    Returns:
        dict[str, np.ndarray] - Очищенные ряды углов (по кадрам).
    """
    out: Dict[str, np.ndarray] = {}
    for name, arr in angles.items():
        x = _deglitch_series(arr, cfg.glitch_thr_deg)
        if name == "trunk_tilt":
            x = _unwrap_deg(x)
        x = _smooth_series(x, cfg.smooth_win, cfg.smooth_poly)
        out[name] = x
    return out


# -------------------------
# 4) Reps -> phase -> reference
# -------------------------
def resample_repeat(
    series: np.ndarray, start: int, end: int, n_points: int
) -> np.ndarray:
    """
    Ресэмплит один повтор [start,end] в фазу 0..100% (n_points значений).

    Returns:
        np.ndarray - Массив длины n_points.
    """
    start = int(start)
    end = int(end)
    end = max(end, start + 1)

    idx = np.arange(start, end + 1)
    y = series[start : end + 1]

    phase_src = (idx - start) / (end - start)
    phase_dst = np.linspace(0.0, 1.0, n_points)
    return np.interp(phase_dst, phase_src, y)


def find_repetitions_by_elbow(
    angles_clean: Dict[str, np.ndarray],
    fps: float,
    cfg: ReferenceBuildConfig,
) -> List[Tuple[int, int]]:
    """
    Находит интервалы повторов по пикам среднего локтевого угла.

    Returns:
        list[tuple[int,int]] - Список (start_frame, end_frame) для каждого повтора.
    """
    elbow_mean = (
        angles_clean["elbow_left"] + angles_clean["elbow_right"]
    ) / 2.0
    min_peak_dist_frames = max(1, int(fps * cfg.min_rep_sec))
    min_rep_len = int(max(6, cfg.min_rep_len_sec * fps))

    try:
        peaks, _ = find_peaks(
            elbow_mean, prominence=cfg.prom_deg, distance=min_peak_dist_frames
        )
    except Exception:
        x = elbow_mean
        cand = np.where(
            (np.r_[True, x[1:] >= x[:-1]] & np.r_[x[:-1] > x[1:], True])
        )[0]
        peaks = []
        last = -(10**9)
        for i in cand:
            if i - last >= min_peak_dist_frames:
                peaks.append(i)
                last = i
        peaks = np.array(peaks, dtype=int)

    reps: List[Tuple[int, int]] = []
    for i in range(len(peaks) - 1):
        s, e = int(peaks[i]), int(peaks[i + 1])
        if e - s >= min_rep_len:
            reps.append((s, e))
    return reps


def build_reference_profiles(
    angles_clean: Dict[str, np.ndarray],
    reps: List[Tuple[int, int]],
    cfg: ReferenceBuildConfig,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Строит эталон mean/std по фазе 0..100% для каждого угла.

    Returns:
        dict - reference[name] = {"mean": np.ndarray, "std": np.ndarray}
    """
    phase_profiles = {name: [] for name in angles_clean.keys()}

    for s, e in reps:
        for name, series in angles_clean.items():
            phase_profiles[name].append(
                resample_repeat(series, s, e, cfg.phase_points)
            )

    reference: Dict[str, Dict[str, np.ndarray]] = {}
    for name, arrs in phase_profiles.items():
        if not arrs:
            continue
        A = np.stack(arrs, axis=0)  # (num_reps, phase_points)
        m = A.mean(axis=0)
        sd = A.std(axis=0, ddof=1) if A.shape[0] > 1 else np.zeros_like(m)
        reference[name] = {"mean": m, "std": sd}
    return reference


def export_reference_to_json(
    reference: Dict[str, Dict[str, np.ndarray]],
    *,
    source_video: str,
    fps: float,
    cfg: ReferenceBuildConfig,
) -> Dict:
    """
    Собирает JSON-совместимую структуру для ideal_reference_profile.json.
    """
    snap_idx = {
        "0": 0,
        "25": cfg.phase_points // 4,
        "50": cfg.phase_points // 2,
        "75": 3 * cfg.phase_points // 4,
        "100": cfg.phase_points - 1,
    }

    data: Dict = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_video": str(source_video),
            "fps": float(fps),
            "phase_points": int(cfg.phase_points),
            "reps_used": int(0),  # заполним снаружи
            "smoothing": {
                "method": "savgol_or_ma",
                "window": int(cfg.smooth_win),
                "poly": int(cfg.smooth_poly),
            },
            "angles": list(reference.keys()),
        }
    }

    for name, stats in reference.items():
        m = stats["mean"]
        sd = stats["std"]
        data[name] = {
            "mean": m.tolist(),
            "std": sd.tolist(),
            "snapshots": {k: float(m[i]) for k, i in snap_idx.items()},
        }
    return data


def build_and_save_reference(
    *,
    video_path: str | Path,
    out_npz_path: str | Path,
    out_json_path: str | Path,
    fps: float,
    cfg: ReferenceBuildConfig = ReferenceBuildConfig(),
    device: str = "cpu",
    save_debug_vis: bool = False,
    debug_dir: str | Path | None = None,
) -> Dict:
    """
    Главная функция: строит эталон и сохраняет артефакты.

    Parameters:
        video_path : str | Path - Видео эталонного выполнения.
        out_npz_path : str | Path - Куда сохранить reference_keypoints.npz (kpts + scores).
        out_json_path : str | Path - Куда сохранить ideal_reference_profile.json (meta + mean/std).
        fps : float - FPS видео (если уже посчитан снаружи).
        cfg : ReferenceBuildConfig - Настройки построения эталона.
        device : str - Устройство для MMPoseInferencer.
        save_debug_vis : bool - Если True — сохранит визуализацию и raw-json предсказаний MMPose.
        debug_dir : str | Path | None - База для debug-вывода. Если None — берётся папка рядом с out_json_path.
    Returns:
        dict - JSON-структура эталона (то, что записано в ideal_reference_profile.json).
    Raises:
        RuntimeError - Если не удалось найти повторы (и нет данных для построения эталона).
    """
    video_path = Path(video_path)
    out_npz_path = Path(out_npz_path)
    out_json_path = Path(out_json_path)

    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    if debug_dir is None:
        debug_dir = out_json_path.parent / "_debug"
    debug_dir = Path(debug_dir)

    vis_dir = debug_dir / "vis_ref" if save_debug_vis else None
    pred_dir = debug_dir / "preds_ref" if save_debug_vis else None
    if save_debug_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

    # 1) keypoints
    kpts, scores = extract_keypoints_from_video(
        video_path, device=device, save_vis_dir=vis_dir, save_pred_dir=pred_dir
    )
    np.savez(out_npz_path, kpts=kpts, scores=scores)

    # 2) normalize
    k_norm, _ = clean_and_normalize_skeleton(kpts, scores)

    # 3) angles
    angles = compute_angles_series(k_norm)
    angles_clean = clean_angles(angles, cfg)

    # 4) reps
    reps = find_repetitions_by_elbow(angles_clean, fps=fps, cfg=cfg)
    if not reps:
        raise RuntimeError(
            "Не удалось выделить повторы на эталонном видео (reps пуст)."
        )

    # 5) build reference
    reference = build_reference_profiles(angles_clean, reps, cfg)

    # 6) export json + save
    ref_json = export_reference_to_json(
        reference, source_video=str(video_path), fps=fps, cfg=cfg
    )
    ref_json["meta"]["reps_used"] = int(len(reps))

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(ref_json, f, ensure_ascii=False, indent=2)

    return ref_json
