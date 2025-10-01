"""Utilities for packaging LangGraph outputs into IBBOT strategy payloads."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

from pydantic import BaseModel, Field


class StrategyConversionError(RuntimeError):
    """Raised when we can't convert graph output into an IBBOT bundle."""


class StrategySignal(BaseModel):
    """Final trading instruction suitable for IBBOT ingestion."""

    symbol: str
    action: str
    quantity: float
    confidence: float | None = None
    rationale: str | None = None
    source_agent: str = Field(description="Agent that produced the decision")
    model_provider: str | None = Field(default=None, description="LLM provider that generated the decision")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RiskDirective(BaseModel):
    """Risk guard-rail extracted from the risk manager."""

    symbol: str
    max_notional: float | None = None
    max_shares: int | None = None
    reference_price: float | None = None
    source_agent: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StrategyBundle(BaseModel):
    """Aggregate payload of signals and risk directives."""

    data_provider: str | None = None
    model_provider: str | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    strategy_mode: str | None = None
    data_timeframe: str | None = None
    workflow_metadata: Dict[str, Any] = Field(default_factory=dict)
    raw_decisions: Dict[str, Any] = Field(default_factory=dict)
    signals: list[StrategySignal] = Field(default_factory=list)
    risk_directives: list[RiskDirective] = Field(default_factory=list)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _safe_json_decode(raw: Any) -> Dict[str, Any]:
    import json

    if raw is None:
        raise StrategyConversionError("Portfolio manager message is empty")

    if isinstance(raw, Mapping):
        return dict(raw)

    if not isinstance(raw, str):
        raise StrategyConversionError("Portfolio manager response is not JSON serialisable")

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StrategyConversionError(f"Invalid JSON in portfolio manager response: {exc}") from exc

    if not isinstance(decoded, Mapping):
        raise StrategyConversionError("Portfolio manager output must be a mapping")

    return dict(decoded)


def _extract_portfolio_message(state: Mapping[str, Any]) -> tuple[str, Any]:
    messages: Iterable[Any] = state.get("messages", []) if isinstance(state, Mapping) else []
    for message in reversed(list(messages)):
        name = getattr(message, "name", None)
        if name and str(name).startswith("portfolio_manager"):
            return str(name), getattr(message, "content", None)
    raise StrategyConversionError("Portfolio manager message not found in state")


def _collect_analyst_inputs(analyst_signals: Mapping[str, Any], symbol: str) -> Dict[str, Any]:
    collected: Dict[str, Any] = {}
    for agent, payload in analyst_signals.items():
        if not isinstance(payload, Mapping):
            continue
        if str(agent).startswith("risk_management_agent"):
            continue
        ticker_payload = payload.get(symbol)
        if not isinstance(ticker_payload, Mapping):
            continue
        collected[str(agent)] = dict(ticker_payload)
    return collected


def _collect_risk_directives(
    analyst_signals: Mapping[str, Any],
    timestamp: datetime,
) -> list[RiskDirective]:
    directives: list[RiskDirective] = []
    for agent, payload in analyst_signals.items():
        if not str(agent).startswith("risk_management_agent"):
            continue
        if not isinstance(payload, Mapping):
            continue
        for symbol, directive in payload.items():
            if not isinstance(directive, Mapping):
                continue
            remaining_limit = _coerce_float(directive.get("remaining_position_limit"))
            reference_price = _coerce_float(directive.get("current_price"))
            max_shares: int | None = None
            if remaining_limit is not None and reference_price and reference_price > 0:
                max_shares = _coerce_int(remaining_limit / reference_price)

            metadata = {
                "volatility": directive.get("volatility_metrics"),
                "correlation": directive.get("correlation_metrics"),
                "reasoning": directive.get("reasoning"),
                "signal_metadata": directive.get("signal_metadata"),
            }

            directives.append(
                RiskDirective(
                    symbol=str(symbol),
                    max_notional=remaining_limit,
                    max_shares=max_shares,
                    reference_price=reference_price,
                    source_agent=str(agent),
                    generated_at=timestamp,
                    metadata={k: v for k, v in metadata.items() if v is not None},
                )
            )
    return directives


def build_strategy_bundle(
    state: Mapping[str, Any],
    decisions: Optional[Mapping[str, Any]] = None,
    *,
    timestamp: datetime | None = None,
) -> StrategyBundle:
    """Convert the LangGraph final state into an IBBOT-ready bundle."""

    if not isinstance(state, Mapping):
        raise StrategyConversionError("Invalid state payload")

    timestamp = timestamp or datetime.now(timezone.utc)

    try:
        portfolio_agent, content = _extract_portfolio_message(state)
    except StrategyConversionError:
        if decisions is None:
            raise
        portfolio_agent = "portfolio_manager"
        content = decisions

    decoded_decisions = dict(decisions) if isinstance(decisions, Mapping) else _safe_json_decode(content)

    data_block = state.get("data", {}) if isinstance(state.get("data"), Mapping) else {}
    metadata_block = state.get("metadata", {}) if isinstance(state.get("metadata"), Mapping) else {}

    analyst_signals = data_block.get("analyst_signals", {}) if isinstance(data_block.get("analyst_signals"), Mapping) else {}
    current_prices = data_block.get("current_prices", {}) if isinstance(data_block.get("current_prices"), Mapping) else {}
    workflow_metadata = data_block.get("workflow_metadata", {}) if isinstance(data_block.get("workflow_metadata"), Mapping) else {}

    data_provider = metadata_block.get("data_provider") or workflow_metadata.get("data_provider")
    model_provider = metadata_block.get("model_provider")
    strategy_mode = metadata_block.get("strategy_mode") or workflow_metadata.get("strategy_mode")
    data_timeframe = metadata_block.get("data_timeframe") or workflow_metadata.get("data_timeframe")

    signals: list[StrategySignal] = []
    for symbol, payload in decoded_decisions.items():
        if not isinstance(payload, Mapping):
            continue
        action = payload.get("action")
        quantity = payload.get("quantity")
        if action is None or quantity is None:
            continue

        confidence = _coerce_float(payload.get("confidence"))
        rationale = payload.get("reasoning")
        analyst_inputs = _collect_analyst_inputs(analyst_signals, symbol)

        metadata = {
            "raw_decision": dict(payload),
            "current_price": _coerce_float(current_prices.get(symbol)),
            "analyst_inputs": analyst_inputs,
            "strategy_mode": strategy_mode,
        }

        signals.append(
            StrategySignal(
                symbol=str(symbol),
                action=str(action),
                quantity=float(quantity),
                confidence=confidence,
                rationale=str(rationale) if rationale is not None else None,
                source_agent=portfolio_agent,
                model_provider=model_provider,
                generated_at=timestamp,
                metadata={k: v for k, v in metadata.items() if v not in (None, {}, [])},
            )
        )

    if not signals:
        raise StrategyConversionError("No portfolio decisions were generated")

    risk_directives = _collect_risk_directives(analyst_signals, timestamp)

    bundle = StrategyBundle(
        data_provider=str(data_provider) if data_provider is not None else None,
        model_provider=str(model_provider) if model_provider is not None else None,
        generated_at=timestamp,
        strategy_mode=str(strategy_mode) if strategy_mode is not None else None,
        data_timeframe=str(data_timeframe) if data_timeframe is not None else None,
        workflow_metadata=dict(workflow_metadata),
        raw_decisions=decoded_decisions,
        signals=signals,
        risk_directives=risk_directives,
    )

    return bundle


def attach_strategy_bundle(
    state: MutableMapping[str, Any],
    decisions: Optional[Mapping[str, Any]] = None,
) -> tuple[Optional[StrategyBundle], Optional[str]]:
    """Attach the bundle to the state and return (bundle, error_message)."""

    try:
        bundle = build_strategy_bundle(state, decisions)
    except StrategyConversionError as exc:
        state.setdefault("data", {}).setdefault("ibbot_strategy", {
            "available": False,
            "error": str(exc),
        })
        return None, str(exc)
    except Exception as exc:  # pragma: no cover - defensive logging
        state.setdefault("data", {}).setdefault("ibbot_strategy", {
            "available": False,
            "error": "unexpected_error",
        })
        raise exc

    state.setdefault("data", {})["ibbot_strategy"] = {
        "available": True,
        "bundle": bundle.model_dump(mode="json"),
    }
    return bundle, None
