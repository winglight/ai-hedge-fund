"""Utilities for converting internal signals into ibbot-compatible payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StrategySignal(BaseModel):
    """Normalized representation of a portfolio action for ibbot."""

    ticker: str
    action: str
    quantity: int = 0
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    source_agent: str = Field(..., alias="sourceAgent")
    provider: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), alias="generatedAt")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        populate_by_name = True


class RiskDirective(BaseModel):
    """Risk guardrails associated with a ticker for ibbot."""

    ticker: str
    remaining_position_limit: float = Field(..., alias="remainingPositionLimit")
    current_price: Optional[float] = Field(default=None, alias="currentPrice")
    provider: str
    source_agent: str = Field(..., alias="sourceAgent")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), alias="generatedAt")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        populate_by_name = True


class StrategyBundle(BaseModel):
    """Complete payload that can be submitted to ibbot."""

    provider: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), alias="generatedAt")
    portfolio_agent: str = Field(..., alias="portfolioAgent")
    signals: List[StrategySignal]
    risk_directives: List[RiskDirective] = Field(default_factory=list, alias="riskDirectives")
    analyst_context: Dict[str, Any] = Field(default_factory=dict, alias="analystContext")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        populate_by_name = True


def _ensure_datetime(value: Optional[datetime]) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_risk_directives(
    analyst_signals: Dict[str, Any],
    *,
    provider: str,
    generated_at: datetime,
) -> List[RiskDirective]:
    directives: List[RiskDirective] = []
    for agent_id, payload in analyst_signals.items():
        if not agent_id.startswith("risk_management"):
            continue
        if not isinstance(payload, dict):
            continue
        for ticker, ticker_payload in payload.items():
            if not isinstance(ticker_payload, dict):
                continue
            remaining_limit = ticker_payload.get("remaining_position_limit")
            current_price = ticker_payload.get("current_price")
            metadata = {
                key: ticker_payload.get(key)
                for key in ("volatility_metrics", "correlation_metrics", "reasoning")
                if ticker_payload.get(key) is not None
            }
            directives.append(
                RiskDirective(
                    ticker=ticker,
                    remaining_position_limit=float(remaining_limit or 0.0),
                    current_price=_safe_float(current_price),
                    provider=provider,
                    source_agent=agent_id,
                    generated_at=generated_at,
                    metadata=metadata,
                )
            )
    return directives


def _collect_strategy_signals(
    decisions: Dict[str, Any],
    *,
    portfolio_agent: str,
    provider: str,
    generated_at: datetime,
) -> List[StrategySignal]:
    signals: List[StrategySignal] = []
    for ticker, payload in decisions.items():
        if not isinstance(payload, dict):
            continue
        action = str(payload.get("action", "")).lower()
        quantity_raw = payload.get("quantity", 0)
        try:
            quantity = int(quantity_raw)
        except (TypeError, ValueError):
            quantity = 0
        confidence = _safe_float(payload.get("confidence"))
        rationale = payload.get("reasoning")
        metadata = {k: v for k, v in payload.items() if k not in {"action", "quantity", "confidence", "reasoning"}}
        signals.append(
            StrategySignal(
                ticker=ticker,
                action=action,
                quantity=quantity,
                confidence=confidence,
                rationale=rationale,
                source_agent=portfolio_agent,
                provider=provider,
                generated_at=generated_at,
                metadata=metadata,
            )
        )
    if not signals:
        raise ValueError("No portfolio decisions available to convert into ibbot strategy signals")
    return signals


def build_strategy_bundle(
    *,
    decisions: Dict[str, Any],
    analyst_signals: Dict[str, Any],
    provider: str,
    portfolio_agent: str,
    generated_at: Optional[datetime] = None,
    context: Optional[Dict[str, Any]] = None,
) -> StrategyBundle:
    """Construct an ibbot-compatible strategy bundle from agent outputs."""

    normalized_generated_at = _ensure_datetime(generated_at)
    signals = _collect_strategy_signals(
        decisions,
        portfolio_agent=portfolio_agent,
        provider=provider,
        generated_at=normalized_generated_at,
    )
    risk_directives = _collect_risk_directives(
        analyst_signals,
        provider=provider,
        generated_at=normalized_generated_at,
    )

    bundle = StrategyBundle(
        provider=provider,
        generated_at=normalized_generated_at,
        portfolio_agent=portfolio_agent,
        signals=signals,
        risk_directives=risk_directives,
        analyst_context={k: v for k, v in analyst_signals.items() if not k.startswith("risk_management")},
        metadata=context or {},
    )
    return bundle

