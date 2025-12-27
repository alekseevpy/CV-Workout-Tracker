"""
mmpose_mvp/gemini_coach_client.py

Клиент для вызова Gemini Developer API (Google AI Studio) и получения текста
рекомендаций по технике на основе промпта, который вы строите в llm_coach.py.

Назначение:
- Принять готовый prompt (строку) и вернуть текст ответа модели.
- Иметь безопасный фолбэк по имени модели (если указанная недоступна).
- Уметь (опционально) логировать/сохранять результат рядом с артефактами.

Зависимости:
- pip install google-genai
- ENV: GEMINI_API_KEY должен быть задан (рекомендуется хранить в окружении, а не в коде).

Примечание по моделям:
- Если "gemini-2.0-flash" недоступна, используем fallback_models по очереди.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from google import genai
from google.genai import types


@dataclass(frozen=True)
class GeminiCallConfig:
    """
    Конфигурация вызова Gemini.

    Parameters:
        model : str
            Основное имя модели (например, "gemini-2.0-flash").
        fallback_models : list[str]
            Фолбэк-модели, если основная недоступна/не найдена.
        temperature : float
            Температура генерации (0.2–0.7 обычно ок).
        max_output_tokens : int
            Ограничение длины ответа.
        top_p : float | None
            Опционально top_p.
    """

    model: str = "gemini-2.0-flash"
    fallback_models: List[str] = (
        "gemini-2.5-flash",
        "gemini-3-flash-preview",
        "gemini-2.0-flash-lite",
    )
    temperature: float = 0.4
    max_output_tokens: int = 500
    top_p: Optional[float] = None


class GeminiCoachClient:
    """
    Обёртка над Google Gen AI SDK для текстового запроса.

    Использование:
        client = GeminiCoachClient()
        text = client.generate_text(prompt)
    """

    def __init__(self, *, api_key: Optional[str] = None) -> None:
        """
        Создаёт клиента.

        Parameters:
            api_key : str | None
                Если None — SDK попробует взять ключ из окружения (GEMINI_API_KEY).
        """
        self._client = genai.Client(api_key=api_key)

    def generate_text(
        self,
        prompt: str,
        *,
        cfg: GeminiCallConfig = GeminiCallConfig(),
    ) -> str:
        """
        Отправляет prompt в Gemini и возвращает текст ответа.

        Parameters:
            prompt : str
                Готовый промпт (например, из llm_coach.build_llm_prompt_bundle()).
            cfg : GeminiCallConfig
                Настройки генерации и список fallback-моделей.

        Returns:
            str
                Текст ответа модели (без форматирования SDK).

        Raises:
            RuntimeError
                Если не удалось вызвать ни одну модель из списка (основная + fallback).
        """
        models_to_try = [cfg.model, *list(cfg.fallback_models)]

        gen_cfg = types.GenerateContentConfig(
            temperature=float(cfg.temperature),
            max_output_tokens=int(cfg.max_output_tokens),
            top_p=(None if cfg.top_p is None else float(cfg.top_p)),
        )

        last_err: Exception | None = None
        for model_name in models_to_try:
            try:
                resp = self._client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=gen_cfg,
                )
                # resp.text — удобное поле SDK
                if getattr(resp, "text", None):
                    return resp.text.strip()
                # на всякий случай:
                return str(resp).strip()
            except Exception as e:  # noqa: BLE001
                last_err = e

        raise RuntimeError(
            "Gemini call failed for all models. "
            f"Tried: {models_to_try}. Last error: {repr(last_err)}"
        ) from last_err


def save_text(text: str, out_path: str | Path) -> Path:
    """
    Сохраняет текст в файл (utf-8).

    Parameters:
        text : str
            Текст для сохранения.
        out_path : str | Path
            Путь до .txt

    Returns:
        Path
            Путь до сохранённого файла.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p
