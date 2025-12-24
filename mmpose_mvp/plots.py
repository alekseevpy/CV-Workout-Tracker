"""
mmpose_mvp/plots.py

Модуль визуализации фазовых профилей углов для MVP.

Что делает модуль:
1) Строит единый график (одно полотно) эталонных профилей:
   - ref_mean ± ref_std для всех углов.
2) Строит единый график сравнения:
   - ref_mean ± ref_std
   - user_mean поверх эталона.
3) Каждый тип визуализации сохраняется в ОДИН PNG.

Важно:
- Все углы рисуются на одном холсте через subplots.
- Ось X — фаза выполнения (0–100%).
- Ось Y — угол в градусах.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _ensure_dir(path: str | Path) -> Path:
    """
    Создаёт директорию (parents=True), если она не существует.

    Parameters:
        path : str | Path

    Returns:
        Path
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _phase_axis(phase_points: int) -> np.ndarray:
    """
    Создаёт ось фаз 0..100%.

    Parameters:
        phase_points : int

    Returns:
        np.ndarray shape (phase_points,)
    """
    return np.linspace(0.0, 100.0, phase_points)


def _load_reference(ref_json_path: str | Path) -> Tuple[dict, Dict[str, dict]]:
    """
    Загружает эталонный JSON.

    Returns:
        meta : dict
        profiles : dict[angle] -> {"mean": np.ndarray, "std": np.ndarray}
    """
    with open(ref_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["meta"]
    profiles = {}

    for a in meta["angles"]:
        profiles[a] = {
            "mean": np.asarray(data[a]["mean"], dtype=float),
            "std": np.asarray(data[a]["std"], dtype=float),
        }

    return meta, profiles


def _load_user_profile(metrics_json_path: str | Path) -> dict:
    """
    Загружает user_profile из last_metrics.json.

    Raises:
        KeyError если user_profile отсутствует.
    """
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "user_profile" not in data:
        raise KeyError(
            "last_metrics.json does not contain 'user_profile'. "
            "User profiles must be saved as mean/std by phase."
        )

    return data["user_profile"]


# ---------------------------------------------------------------------
# Reference plot (single canvas)
# ---------------------------------------------------------------------
def save_reference_profiles_canvas(
    *,
    ref_json_path: str | Path,
    out_png: str | Path,
    band_sigma: float = 1.0,
    dpi: int = 150,
) -> str:
    """
    Строит ОДИН график со всеми эталонными углами (subplots).

    Parameters:
        ref_json_path : str | Path
            Путь к ideal_reference_profile.json.
        out_png : str | Path
            Куда сохранить PNG.
        band_sigma : float
            ref_mean ± band_sigma * ref_std.
        dpi : int

    Returns:
        str
            Абсолютный путь к сохранённому PNG.
    """
    out_png = Path(out_png)
    _ensure_dir(out_png.parent)

    meta, profiles = _load_reference(ref_json_path)
    phase_points = int(meta["phase_points"])
    angles = meta["angles"]

    x = _phase_axis(phase_points)

    n = len(angles)
    fig, axes = plt.subplots(
        nrows=n, ncols=1, figsize=(10, 2.2 * n), sharex=True
    )

    if n == 1:
        axes = [axes]

    for ax, a in zip(axes, angles):
        m = profiles[a]["mean"]
        s = profiles[a]["std"]

        ax.fill_between(
            x,
            m - band_sigma * s,
            m + band_sigma * s,
            alpha=0.25,
            label="ref ± σ",
        )
        ax.plot(x, m, linewidth=2, label="ref mean")
        ax.set_title(a)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("phase (%)")
    fig.suptitle("Reference phase profiles", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(out_png, dpi=dpi)
    plt.close(fig)

    return str(out_png.resolve())


# ---------------------------------------------------------------------
# Comparison plot (single canvas)
# ---------------------------------------------------------------------
def save_comparison_profiles_canvas(
    *,
    ref_json_path: str | Path,
    user_metrics_json_path: str | Path,
    out_png: str | Path,
    band_sigma: float = 1.0,
    dpi: int = 150,
) -> str:
    """
    Строит ОДИН график сравнения (user vs reference) по всем углам.

    Parameters:
        ref_json_path : str | Path
            Путь к ideal_reference_profile.json.
        user_metrics_json_path : str | Path
            Путь к last_metrics.json (должен содержать user_profile).
        out_png : str | Path
            Куда сохранить PNG.
        band_sigma : float
            ref_mean ± band_sigma * ref_std.
        dpi : int

    Returns:
        str
            Абсолютный путь к сохранённому PNG.
    """
    out_png = Path(out_png)
    _ensure_dir(out_png.parent)

    meta, ref_profiles = _load_reference(ref_json_path)
    user_profile = _load_user_profile(user_metrics_json_path)

    phase_points = int(meta["phase_points"])
    angles = meta["angles"]
    x = _phase_axis(phase_points)

    fig, axes = plt.subplots(
        nrows=len(angles),
        ncols=1,
        figsize=(10, 2.2 * len(angles)),
        sharex=True,
    )

    if len(angles) == 1:
        axes = [axes]

    for ax, a in zip(axes, angles):
        if a not in user_profile:
            ax.set_title(f"{a} (no user data)")
            continue

        ref_m = ref_profiles[a]["mean"]
        ref_s = ref_profiles[a]["std"]
        usr_m = np.asarray(user_profile[a]["mean"], dtype=float)

        ax.fill_between(
            x,
            ref_m - band_sigma * ref_s,
            ref_m + band_sigma * ref_s,
            alpha=0.25,
            label="ref ± σ",
        )
        ax.plot(x, ref_m, linewidth=2, label="ref mean")
        ax.plot(x, usr_m, linewidth=2, label="user mean")
        ax.set_title(a)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("phase (%)")
    fig.suptitle("User vs Reference comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(out_png, dpi=dpi)
    plt.close(fig)

    return str(out_png.resolve())
