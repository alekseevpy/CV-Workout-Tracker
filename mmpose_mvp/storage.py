"""
mmpose_mvp/storage.py

Модуль отвечает за файловую организацию данных MVP-трекера упражнений.

Задачи модуля:
- нормализация названий упражнений (RU / EN -> каноническое имя);
- создание и управление файловой структурой для упражнений;
- сохранение загруженных пользователем видео (эталонных и пользовательских);
- сохранение JSON-файлов с результатами обработки и метриками;
- проверка наличия эталонного выполнения упражнения.

Модуль не содержит логики анализа, позы или ML —
он только управляет путями и сохранением данных на диске.

Ожидаемая структура директорий:

mmpose_mvp/
    data_mvp/
    pushups/
        reference/
            reference.mp4
            reference_keypoints.npz
            ideal_reference_profile.json
        user/
            last_attempt.mp4
            last_attempt_keypoints.npz
            last_metrics.json
    ...
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# Поддерживаемые упражнения и их канонические имена
SUPPORTED_EXERCISES = {
    "pushups": "pushups",
    "pullups": "pullups",
    "squats": "squats",
    "abs": "abs",
    "отжимание": "pushups",
    "подтягивание": "pullups",
    "приседание": "squats",
    "пресс": "abs",
}


def normalize_exercise_name(exercise: str) -> str:
    """
    Приводит ввод пользователя к каноническому имени упражнения.
    Поддерживает русские и английские варианты.

    Parameters:
        exercise : str - Название упражнения, введённое пользователем.
    Returns:
        str - Каноническое имя упражнения (например: "pushups").
    Raises:
        ValueError - Если строка пустая или упражнение не поддерживается.
    """
    if not exercise or not exercise.strip():
        raise ValueError("exercise is empty")
    key = exercise.strip().lower()
    if key not in SUPPORTED_EXERCISES:
        raise ValueError(
            f"Unsupported exercise '{exercise}'. ",
            f"Supported: {sorted(set(SUPPORTED_EXERCISES.keys()))}",
        )
    return SUPPORTED_EXERCISES[key]


@dataclass(frozen=True)
class ExercisePaths:
    """
    Контейнер со всеми путями для одного упражнения.

    Используется как единая точка доступа ко всем файлам,
    связанным с конкретным упражнением.
    """

    base: Path

    reference_dir: Path
    user_dir: Path

    reference_video: Path  # reference/reference.mp4
    reference_json: Path  # reference/ideal_reference_profile.json
    reference_kpts_npz: Path  # reference/reference_keypoints.npz

    user_video: Path  # user/last_attempt.mp4
    user_kpts_npz: Path  # user/last_attempt_keypoints.npz
    user_metrics_json: Path  # user/last_metrics.json


def get_exercise_paths(
    base_dir: str | os.PathLike, exercise: str
) -> ExercisePaths:
    """
    Создаёт директории для упражнения (если их ещё нет)
    и возвращает объект со всеми путями.

    Parameters:
        base_dir : str | PathLike - Базовая директория для хранения данных MVP.
        exercise : str - Название упражнения (RU или EN).
    Returns:
        ExercisePaths - Объект со всеми путями для эталона и пользовательских данных.
    """
    ex = normalize_exercise_name(exercise)
    base = Path(base_dir).expanduser().resolve() / ex

    reference_dir = base / "reference"
    user_dir = base / "user"
    reference_dir.mkdir(parents=True, exist_ok=True)
    user_dir.mkdir(parents=True, exist_ok=True)

    return ExercisePaths(
        base=base,
        reference_dir=reference_dir,
        user_dir=user_dir,
        reference_video=reference_dir / "reference.mp4",
        reference_json=reference_dir / "ideal_reference_profile.json",
        reference_kpts_npz=reference_dir / "reference_keypoints.npz",
        user_video=user_dir / "last_attempt.mp4",
        user_kpts_npz=user_dir / "last_attempt_keypoints.npz",
        user_metrics_json=user_dir / "last_metrics.json",
    )


def save_video_bytes(video_bytes: bytes, out_path: str | os.PathLike) -> str:
    """
    Сохраняет видео, переданное как набор байтов.
    Используется для сохранения файлов, загруженных через UI (Streamlit uploader).
    Файл всегда перезаписывается.

    Parameters:
        video_bytes : bytes - Бинарное содержимое видеофайла.
        out_path : str | PathLike - Путь, куда сохранить видео.
    Returns:
        str - Абсолютный путь к сохранённому файлу.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(video_bytes)
    return str(out_path.resolve())


def save_json(obj: dict, out_path: str | os.PathLike) -> str:
    """
    Сохраняет Python-словарь в JSON-файл. Файл всегда перезаписывается.

    Parameters:
        obj : dict - Объект для сериализации в JSON.
        out_path : str | PathLike - Путь для сохранения JSON.
    Returns:
        str - Абсолютный путь к сохранённому файлу.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return str(out_path.resolve())


def ensure_reference_exists(paths: ExercisePaths) -> None:
    """
    Проверяет, что эталон для упражнения существует.
    Используется перед анализом пользовательского видео.

    Parameters:
        paths : ExercisePaths - Пути, полученные через get_exercise_paths.

    Raises:
        FileNotFoundError - Если эталонное видео или эталонный JSON отсутствуют.
    """
    if not paths.reference_json.exists():
        raise FileNotFoundError(
            f"Reference JSON not found: {paths.reference_json}"
        )
    if not paths.reference_video.exists():
        raise FileNotFoundError(
            f"Reference video not found: {paths.reference_video}"
        )
