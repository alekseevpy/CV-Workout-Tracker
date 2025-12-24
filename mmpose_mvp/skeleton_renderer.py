"""
mmpose_mvp/skeleton_renderer.py

Модуль визуализации нормализованного скелета (COCO-17) в видеоформате.

Назначение:
1) Отрисовка выполнения упражнения в виде "скелетика" ПОСЛЕ нормализации
   (центрирование + масштабирование), чтобы форма движения была сопоставима
   между разными людьми и видео.
2) Поддержка:
   - одиночного видео (один скелет),
   - split-screen видео (эталон | пользователь).
3) Автоматическая подгонка масштаба и центра (auto-fit), чтобы скелет
   всегда полностью помещался в кадр и не "улетал".

Ключевая идея:
- Мы НЕ рисуем в координатах камеры.
- Мы рисуем в нормализованном пространстве (k_norm), используем авто-fit
  для перевода world -> pixel.

Используется для:
- отладки качества детекции,
- визуального объяснения графиков углов,
- демонстрации MVP (наглядно видно "как двигается").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

# -------------------------
# COCO-17 skeleton
# -------------------------

COCO_EDGES: List[Tuple[int, int]] = [
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (5, 6),  # shoulders
    (11, 12),  # hips
    (5, 11),
    (6, 12),  # torso
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
]


# -------------------------
# Configs
# -------------------------


@dataclass(frozen=True)
class NormalizedRenderConfig:
    """
    Конфигурация рендера нормализованного скелета.

    Parameters:
        canvas_w : int
            Ширина кадра в пикселях.
        canvas_h : int
            Высота кадра в пикселях.
        world_scale : float
            Масштаб world -> pixel.
        center_xy : tuple[int, int]
            Смещение (pixel), куда попадает (0,0) world.
        flip_y : bool
            Инвертировать ось Y (обычно True для экранных координат).
        point_radius : int
            Радиус точек суставов.
        line_thickness : int
            Толщина линий костей.
        draw_axes : bool
            Рисовать оси координат (редко нужно).
    """

    canvas_w: int
    canvas_h: int
    world_scale: float
    center_xy: Tuple[int, int]
    flip_y: bool = True
    point_radius: int = 4
    line_thickness: int = 2
    draw_axes: bool = False


@dataclass(frozen=True)
class SplitScreenConfig:
    """
    Конфигурация split-screen видео.

    Parameters:
        panel_w : int
            Ширина одной панели.
        panel_h : int
            Высота панели.
        gap_px : int
            Расстояние между панелями.
        bg_color : tuple[int,int,int]
            Цвет фона (BGR).
    """

    panel_w: int = 480
    panel_h: int = 480
    gap_px: int = 10
    bg_color: Tuple[int, int, int] = (20, 20, 20)


# -------------------------
# Auto-fit helper
# -------------------------


def auto_render_config_from_k_norm(
    k_norm: np.ndarray,
    *,
    canvas_w: int,
    canvas_h: int,
    margin_px: int = 40,
    flip_y: bool = True,
    quantile: float = 0.98,
) -> NormalizedRenderConfig:
    """
    Автоматически подбирает масштаб и центр, чтобы скелет влезал в canvas.

    Parameters:
        k_norm : np.ndarray
            Нормализованные координаты (T,17,2).
        canvas_w, canvas_h : int
            Размеры холста.
        margin_px : int
            Отступ от краёв.
        flip_y : bool
            Инвертировать Y.
        quantile : float
            Квантиль для подавления выбросов.

    Returns:
        NormalizedRenderConfig
    """
    pts = k_norm.reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]

    if len(pts) < 10:
        return NormalizedRenderConfig(
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            world_scale=min(canvas_w, canvas_h) / 4.0,
            center_xy=(canvas_w // 2, canvas_h // 2),
            flip_y=flip_y,
        )

    x = pts[:, 0]
    y = pts[:, 1]

    q = float(quantile)
    x_min, x_max = np.quantile(x, [1 - q, q])
    y_min, y_max = np.quantile(y, [1 - q, q])

    w_world = max(1e-6, x_max - x_min)
    h_world = max(1e-6, y_max - y_min)

    avail_w = max(1, canvas_w - 2 * margin_px)
    avail_h = max(1, canvas_h - 2 * margin_px)

    scale = min(avail_w / w_world, avail_h / h_world)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    px = canvas_w / 2 - cx * scale
    py = canvas_h / 2 + cy * scale if flip_y else canvas_h / 2 - cy * scale

    return NormalizedRenderConfig(
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        world_scale=float(scale),
        center_xy=(int(px), int(py)),
        flip_y=flip_y,
    )


# -------------------------
# Drawing primitives
# -------------------------


def world_to_pixel(
    xy: np.ndarray,
    cfg: NormalizedRenderConfig,
) -> Tuple[int, int]:
    """
    Перевод world -> pixel координат.
    """
    x, y = float(xy[0]), float(xy[1])
    px = cfg.center_xy[0] + x * cfg.world_scale
    py = (
        cfg.center_xy[1] - y * cfg.world_scale
        if cfg.flip_y
        else cfg.center_xy[1] + y * cfg.world_scale
    )
    return int(round(px)), int(round(py))


def draw_normalized_skeleton_frame(
    frame: np.ndarray,
    kpts: np.ndarray,
    scores: np.ndarray | None,
    cfg: NormalizedRenderConfig,
    *,
    min_score: float = 0.2,
    color_points: Tuple[int, int, int] = (0, 220, 255),
    color_lines: Tuple[int, int, int] = (0, 180, 255),
) -> np.ndarray:
    """
    Рисует один кадр нормализованного скелета.

    Parameters:
        frame : np.ndarray
            BGR изображение.
        kpts : np.ndarray
            (17,2) координаты.
        scores : np.ndarray | None
            (17,) confidence или None.
        cfg : NormalizedRenderConfig
            Конфиг рендера.
        min_score : float
            Минимальный score для отрисовки точки.
    """
    pts_px: List[Tuple[int, int] | None] = []

    for i in range(17):
        if scores is not None and scores[i] < min_score:
            pts_px.append(None)
        else:
            pts_px.append(world_to_pixel(kpts[i], cfg))

    for a, b in COCO_EDGES:
        if pts_px[a] is not None and pts_px[b] is not None:
            cv2.line(
                frame, pts_px[a], pts_px[b], color_lines, cfg.line_thickness
            )

    for p in pts_px:
        if p is not None:
            cv2.circle(frame, p, cfg.point_radius, color_points, -1)

    return frame


# -------------------------
# Video renderers
# -------------------------


def render_normalized_skeleton_video(
    *,
    k_norm: np.ndarray,
    scores: np.ndarray | None,
    out_path: str | Path,
    fps: float,
    canvas_w: int = 480,
    canvas_h: int = 480,
    min_score: float = 0.2,
) -> Path:
    """
    Рендер одного нормализованного скелета в видео.

    Parameters:
        k_norm : np.ndarray
            (T,17,2)
        scores : np.ndarray | None
            (T,17)
        out_path : str | Path
            Куда сохранить mp4.
        fps : float
            FPS видео.
    """
    cfg = auto_render_config_from_k_norm(
        k_norm,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (canvas_w, canvas_h),
    )

    T = k_norm.shape[0]
    for t in range(T):
        frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        frame = draw_normalized_skeleton_frame(
            frame,
            k_norm[t],
            None if scores is None else scores[t],
            cfg,
            min_score=min_score,
        )
        writer.write(frame)

    writer.release()
    return out_path


def render_split_screen_normalized_video(
    *,
    k_norm_left: np.ndarray,
    scores_left: np.ndarray | None,
    k_norm_right: np.ndarray,
    scores_right: np.ndarray | None,
    out_path: str | Path,
    fps: float,
    align: str = "truncate",
    min_score: float = 0.2,
    cfg: SplitScreenConfig = SplitScreenConfig(),
) -> Path:
    """
    Рендер split-screen видео: эталон | пользователь.

    Parameters:
        k_norm_left, k_norm_right : np.ndarray
            (T,17,2)
        scores_left, scores_right : np.ndarray | None
            (T,17)
        align : {"truncate","pad"}
            Как выравнивать длины.
    """
    T1, T2 = k_norm_left.shape[0], k_norm_right.shape[0]
    if align == "truncate":
        T = min(T1, T2)
    else:
        T = max(T1, T2)

    cfg_left = auto_render_config_from_k_norm(
        k_norm_left,
        canvas_w=cfg.panel_w,
        canvas_h=cfg.panel_h,
    )
    cfg_right = auto_render_config_from_k_norm(
        k_norm_right,
        canvas_w=cfg.panel_w,
        canvas_h=cfg.panel_h,
    )

    W = cfg.panel_w * 2 + cfg.gap_px
    H = cfg.panel_h

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )

    for t in range(T):
        frame = np.full((H, W, 3), cfg.bg_color, dtype=np.uint8)

        i1 = min(t, T1 - 1)
        i2 = min(t, T2 - 1)

        left = np.zeros((cfg.panel_h, cfg.panel_w, 3), dtype=np.uint8)
        right = np.zeros((cfg.panel_h, cfg.panel_w, 3), dtype=np.uint8)

        draw_normalized_skeleton_frame(
            left,
            k_norm_left[i1],
            None if scores_left is None else scores_left[i1],
            cfg_left,
            min_score=min_score,
        )
        draw_normalized_skeleton_frame(
            right,
            k_norm_right[i2],
            None if scores_right is None else scores_right[i2],
            cfg_right,
            min_score=min_score,
        )

        frame[:, : cfg.panel_w] = left
        frame[:, cfg.panel_w + cfg.gap_px :] = right

        writer.write(frame)

    writer.release()
    return out_path
