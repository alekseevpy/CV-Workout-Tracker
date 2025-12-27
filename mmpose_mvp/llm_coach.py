"""
mmpose_mvp/llm_coach.py

Модуль генерации текстовых рекомендаций (coach feedback) на основе артефактов сравнения
пользовательского выполнения с эталоном (last_metrics.json).

Идея "Вариант B" (рекомендуемый для MVP):
1) Детерминированно ранжируем "проблемные" углы по метрикам (mae, pct_outside_1sigma).
2) Формируем компактный контекст (top-K углов + ключевые числа) и ограничения.
3) Отдаём контекст в LLM, чтобы он:
   - сформулировал понятные рекомендации,
   - не "выдумывал" причины, а опирался на метрики,
   - следовал правилам (например, trunk_tilt может быть wrap issue).

Почему так лучше:
- меньше галлюцинаций,
- проще контролировать приоритеты,
- проще дебажить (видно, какие углы выбраны и почему).

Вход:
- last_metrics.json (результат user_analyzer.analyze_user_video_and_compare),
  где есть:
  - meta_user, meta_reference
  - summary
  - metrics_by_angle
  - user_profile (mean/std по фазе для каждого угла)

Выход:
- текстовый промпт для LLM,
- (опционально) структура "facts" для логирования/отладки,
- функции для безопасного ранжирования и правил (включая trunk_tilt wrap caution).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -------------------------
# Data models
# -------------------------


@dataclass(frozen=True)
class AngleMetric:
    """
    Метрики по одному углу из блока metrics_by_angle.

    Parameters:
        mae : float
            Mean Absolute Error между user_mean и ref_mean по фазе.
        max_err : float
            Максимальная абсолютная ошибка по фазе.
        pct_outside_1sigma : float
            % фазовых точек, где |user_mean - ref_mean| > ref_std.
    """

    mae: float
    max_err: float
    pct_outside_1sigma: float


@dataclass(frozen=True)
class RankedAngle:
    """
    Результат ранжирования угла.

    Parameters:
        angle : str
            Имя угла (например, elbow_left).
        score : float
            Итоговый score (чем больше — тем проблемнее).
        metric : AngleMetric
            Исходные метрики угла.
        reasons : list[str]
            Короткие причины/объяснения, почему угол попал в топ.
    """

    angle: str
    score: float
    metric: AngleMetric
    reasons: List[str]


@dataclass(frozen=True)
class PromptBundle:
    """
    Результат подготовки промпта для LLM.

    Parameters:
        prompt : str
            Готовый промпт, который можно отправлять в LLM.
        facts : dict
            Структура фактов/аргументов (для логирования, воспроизводимости и дебага).
    """

    prompt: str
    facts: Dict[str, Any]


# -------------------------
# IO helpers
# -------------------------


def load_last_metrics(path: str | Path) -> Dict[str, Any]:
    """
    Загружает last_metrics.json.

    Parameters:
        path : str | Path
            Путь к last_metrics.json.

    Returns:
        dict
            Содержимое файла как Python-словарь.

    Raises:
        FileNotFoundError
            Если файл не найден.
        json.JSONDecodeError
            Если файл битый/невалидный JSON.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"last_metrics.json not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# Ranking logic (Variant B)
# -------------------------


def _metric_from_json(m: Dict[str, Any]) -> AngleMetric:
    """
    Преобразует metrics_by_angle[angle] -> AngleMetric.

    Parameters:
        m : dict
            {"mae":..., "max_err":..., "pct_outside_1sigma":...}

    Returns:
        AngleMetric
    """
    return AngleMetric(
        mae=float(m.get("mae", 0.0)),
        max_err=float(m.get("max_err", 0.0)),
        pct_outside_1sigma=float(m.get("pct_outside_1sigma", 0.0)),
    )


def rank_angles(
    metrics_by_angle: Dict[str, Dict[str, Any]],
    *,
    weights: Tuple[float, float] = (1.0, 0.6),
    exclude: Optional[List[str]] = None,
) -> List[RankedAngle]:
    """
    Ранжирует углы по "проблемности".

    Скоринг (простой и объяснимый):
      score = w_mae * mae + w_out * (pct_outside_1sigma)

    Здесь pct_outside_1sigma в процентах (0..100), поэтому ему обычно дают меньший вес.

    Parameters:
        metrics_by_angle : dict
            Блок из last_metrics.json: angle -> {"mae","max_err","pct_outside_1sigma"}.
        weights : tuple[float,float]
            (w_mae, w_out) веса для mae и pct_outside_1sigma.
        exclude : list[str] | None
            Список углов, которые надо исключить (например ["trunk_tilt"]).

    Returns:
        list[RankedAngle]
            Отсортировано по score убыванию.
    """
    exclude_set = set(exclude or [])
    w_mae, w_out = float(weights[0]), float(weights[1])

    ranked: List[RankedAngle] = []
    for angle, m in metrics_by_angle.items():
        if angle in exclude_set:
            continue

        metric = _metric_from_json(m)
        score = w_mae * metric.mae + w_out * metric.pct_outside_1sigma

        reasons = [
            f"mae={metric.mae:.2f}",
            f"pct_outside_1sigma={metric.pct_outside_1sigma:.1f}%",
            f"max_err={metric.max_err:.2f}",
        ]
        ranked.append(
            RankedAngle(
                angle=angle, score=float(score), metric=metric, reasons=reasons
            )
        )

    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked


def apply_trunk_tilt_caution(
    ranked: List[RankedAngle],
    metrics_by_angle: Dict[str, Dict[str, Any]],
    *,
    min_supporting_angles: int = 2,
    supporting_threshold_outside: float = 55.0,
) -> Tuple[List[RankedAngle], Dict[str, Any]]:
    """
    Правило для trunk_tilt: trunk_tilt может быть "wrap issue".

    Суть:
    - trunk_tilt НЕ должен становиться рекомендацией №1 "сам по себе".
    - Мы либо:
      A) понижаем trunk_tilt в ранжировании (если нет подтверждения),
      B) или помечаем его как "сомнительный / требует подтверждения".

    Упрощённая логика "подтверждения":
    - считаем, сколько других углов имеют pct_outside_1sigma >= supporting_threshold_outside
    - если таких углов < min_supporting_angles, то trunk_tilt помечаем как caution

    Parameters:
        ranked : list[RankedAngle]
            Отранжированные углы (без фильтрации).
        metrics_by_angle : dict
            Исходные метрики.
        min_supporting_angles : int
            Минимум других углов с плохим outside, чтобы trunk_tilt считать "подтверждённым".
        supporting_threshold_outside : float
            Порог для подтверждающих углов (в %).

    Returns:
        (ranked_updated, trunk_rule_info)
            ranked_updated : list[RankedAngle]
                Возможно с изменённым порядком (trunk_tilt опущен вниз).
            trunk_rule_info : dict
                Информация о применённом правиле (для логов/промпта).
    """
    trunk = next((x for x in ranked if x.angle == "trunk_tilt"), None)
    if trunk is None:
        return ranked, {"enabled": True, "trunk_present": False}

    support = 0
    for a, m in metrics_by_angle.items():
        if a == "trunk_tilt":
            continue
        out = float(m.get("pct_outside_1sigma", 0.0))
        if out >= supporting_threshold_outside:
            support += 1

    info = {
        "enabled": True,
        "trunk_present": True,
        "supporting_angles_count": int(support),
        "min_supporting_angles": int(min_supporting_angles),
        "supporting_threshold_outside": float(supporting_threshold_outside),
        "action": "none",
    }

    if support < min_supporting_angles:
        # Понижаем trunk_tilt: убираем и добавляем в конец топа (мягкая политика).
        without = [x for x in ranked if x.angle != "trunk_tilt"]
        ranked2 = without + [trunk]
        info["action"] = "demote_trunk_tilt"
        info["note"] = (
            "trunk_tilt treated as potential wrap issue (insufficient support)"
        )
        return ranked2, info

    info["action"] = "keep_trunk_tilt"
    info["note"] = "trunk_tilt supported by other problematic angles"
    return ranked, info


# -------------------------
# Prompt building
# -------------------------


def build_llm_prompt_bundle(
    last_metrics: Dict[str, Any],
    *,
    top_k: int = 6,
    weights: Tuple[float, float] = (1.0, 0.6),
    trunk_tilt_wrap_rule: bool = True,
    min_recommendations: int = 4,
    max_recommendations: int = 6,
) -> PromptBundle:
    """
    Формирует готовый промпт для LLM (Variant B).

    Делает:
    1) Берёт metrics_by_angle, ранжирует углы.
    2) Применяет правило trunk_tilt (если включено).
    3) Берёт top_k углов (после правила) и формирует "facts" + текст промпта.

    Parameters:
        last_metrics : dict
            Загруженный last_metrics.json.
        top_k : int
            Сколько углов отдаём в LLM как основные проблемы.
        weights : tuple[float,float]
            Веса для ранжирования.
        trunk_tilt_wrap_rule : bool
            Включать ли правило "trunk_tilt может быть wrap issue".
        min_recommendations : int
            Минимум рекомендаций, которые просим у LLM.
        max_recommendations : int
            Максимум рекомендаций, которые просим у LLM.

    Returns:
        PromptBundle
            prompt + facts для логирования.
    """
    metrics_by_angle = last_metrics.get("metrics_by_angle", {}) or {}
    meta_user = last_metrics.get("meta_user", {}) or {}
    meta_ref = last_metrics.get("meta_reference", {}) or {}
    summary = last_metrics.get("summary", {}) or {}

    ranked = rank_angles(metrics_by_angle, weights=weights)
    trunk_info: Dict[str, Any] = {"enabled": False}
    if trunk_tilt_wrap_rule:
        ranked, trunk_info = apply_trunk_tilt_caution(ranked, metrics_by_angle)

    top_k = int(top_k)
    top = ranked[: max(1, top_k)]

    min_recommendations = int(min_recommendations)
    max_recommendations = int(max_recommendations)
    if min_recommendations < 1:
        min_recommendations = 1
    if max_recommendations < min_recommendations:
        max_recommendations = min_recommendations

    # компактный "facts" — чтобы LLM не гадал
    facts = {
        "meta_user": {
            "fps": meta_user.get("fps"),
            "reps_found": meta_user.get("reps_found"),
            "reps_used": meta_user.get("reps_used"),
            "warning": meta_user.get("warning"),
        },
        "meta_reference": {
            "fps": meta_ref.get("fps"),
            "phase_points": meta_ref.get("phase_points"),
            "angles": meta_ref.get("angles"),
        },
        "summary": summary,
        "ranking": [
            {
                "angle": x.angle,
                "score": x.score,
                "mae": x.metric.mae,
                "max_err": x.metric.max_err,
                "pct_outside_1sigma": x.metric.pct_outside_1sigma,
                "reasons": x.reasons,
            }
            for x in top
        ],
        "trunk_tilt_rule": trunk_info,
        "constraints": {
            "do_not_overprioritize_trunk_tilt_without_support": True,
            "recommendations_should_reference_metrics": True,
            "min_recommendations": min_recommendations,
            "max_recommendations": max_recommendations,
            "top_problem_angles_in_context": top_k,
        },
    }

    prompt = _format_prompt(facts)
    return PromptBundle(prompt=prompt, facts=facts)


def _format_prompt(facts: Dict[str, Any]) -> str:
    """
    Собирает человекочитаемый промпт из facts.

    Parameters:
        facts : dict
            Структура фактов из build_llm_prompt_bundle().

    Returns:
        str
            Готовый промпт для LLM.
    """
    ranking = facts["ranking"]
    trunk_rule = facts.get("trunk_tilt_rule", {})
    constraints = facts.get("constraints", {})

    min_recs = int(constraints.get("min_recommendations", 4))
    max_recs = int(constraints.get("max_recommendations", 6))

    lines: List[str] = []
    lines.append("Ты — тренер по технике упражнения (отжимания/похожее).")
    lines.append(
        "У тебя есть метрики сравнения пользователя с эталоном по фазе 0..100%."
    )
    lines.append("")
    lines.append("ЗАДАЧА:")
    lines.append(
        f"- Дай {min_recs}–{max_recs} конкретных рекомендаций по технике."
    )
    lines.append(
        "- Каждую рекомендацию обоснуй числами (mae / pct_outside_1sigma / max_err) "
        "для соответствующих углов."
    )
    lines.append("- Не выдумывай причины, опирайся только на метрики.")
    lines.append(
        "- Не дублируй смысл: каждая рекомендация должна быть про разную проблему."
    )
    lines.append("")
    lines.append("ВАЖНОЕ ПРАВИЛО:")
    if constraints.get(
        "do_not_overprioritize_trunk_tilt_without_support", False
    ):
        lines.append(
            "- trunk_tilt может быть 'wrap issue'. НЕ делай trunk_tilt рекомендацией №1, "
            "если это не подтверждается другими углами."
        )
        if trunk_rule.get("enabled"):
            lines.append(
                f"- trunk_tilt_rule_action: {trunk_rule.get('action')}"
            )
            if trunk_rule.get("note"):
                lines.append(
                    f"- trunk_tilt_rule_note: {trunk_rule.get('note')}"
                )
    lines.append("")
    lines.append("КОНТЕКСТ (топ проблемные углы):")
    for i, r in enumerate(ranking, 1):
        lines.append(
            f"{i}) {r['angle']}: mae={r['mae']:.2f}, "
            f"pct_outside_1sigma={r['pct_outside_1sigma']:.1f}%, "
            f"max_err={r['max_err']:.2f}"
        )
    lines.append("")
    lines.append("ФОРМАТ ОТВЕТА:")
    lines.append(
        f"Сформируй {min_recs}–{max_recs} пунктов в формате ниже. "
        "Если после 4–5 пунктов новые советы становятся повтором или водой — остановись."
    )
    lines.append("1) Рекомендация (1–2 предложения)")
    lines.append("   Обоснование: ... (с числами)")
    lines.append("2) ...")
    lines.append("3) ...")
    lines.append("4) ...")
    lines.append("5) ... (если есть смысл)")
    lines.append("6) ... (если есть смысл)")
    lines.append("")
    return "\n".join(lines)
