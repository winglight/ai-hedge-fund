import math
import re

from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.api_key import get_api_key_from_state
import json
import pandas as pd
import numpy as np

from src.tools.api import get_prices, get_intraday_prices, prices_to_df
from src.utils.progress import progress


def safe_float(value, default=0.0):
    """
    Safely convert a value to float, handling NaN cases
    
    Args:
        value: The value to convert (can be pandas scalar, numpy value, etc.)
        default: Default value to return if the input is NaN or invalid
    
    Returns:
        float: The converted value or default if NaN/invalid
    """
    try:
        if pd.isna(value) or np.isnan(value):
            return default
        return float(value)
    except (ValueError, TypeError, OverflowError):
        return default


##### Technical Analyst #####
def technical_analyst_agent(state: AgentState, agent_id: str = "technical_analyst_agent"):
    """
    Sophisticated technical analysis system that combines multiple trading strategies for multiple tickers:
    1. Trend Following
    2. Mean Reversion
    3. Momentum
    4. Volatility Analysis
    5. Statistical Arbitrage Signals
    """
    data = state["data"]
    metadata = state.get("metadata", {})
    workflow_metadata = data.get("workflow_metadata", {}) or {}
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    strategy_mode = (
        metadata.get("strategy_mode")
        or workflow_metadata.get("strategy_mode")
        or metadata.get("strategy", {}).get("mode")
    )
    data_timeframe = metadata.get("data_timeframe") or workflow_metadata.get("data_timeframe")
    data_provider = metadata.get("data_provider") or workflow_metadata.get("data_provider")
    data_granularity = (
        "intraday"
        if is_intraday_mode(strategy_mode, data_timeframe)
        else "end_of_day"
    )

    progress_context = {
        "data_provider": data_provider,
        "strategy_mode": strategy_mode,
        "data_timeframe": data_timeframe,
        "data_granularity": data_granularity,
    }

    technical_analysis: dict[str, dict] = {}
    tickers_to_analyze = tickers[:1] if data_granularity == "intraday" and tickers else tickers

    for ticker in tickers_to_analyze:
        progress.update_status(agent_id, ticker, "Analyzing price data", context=progress_context)

        if data_granularity == "intraday":
            progress.update_status(agent_id, ticker, "Fetching intraday data", context=progress_context)
            prices = get_intraday_prices(
                ticker,
                start_date,
                end_date,
                api_key=api_key,
                provider=data_provider,
            )
            if not prices:
                progress.update_status(
                    agent_id,
                    ticker,
                    "Fallback to end-of-day data",
                    context=progress_context,
                )
                prices = get_prices(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    api_key=api_key,
                )

            if not prices:
                progress.update_status(agent_id, ticker, "Failed: No price data found", context=progress_context)
                continue

            prices_df = prices_to_df(prices)
            if prices_df.empty or len(prices_df) < 5:
                progress.update_status(agent_id, ticker, "Insufficient intraday data", context=progress_context)
                continue

            prices_df = prices_df.sort_values("time")
            intraday_result = analyze_intraday_signals(prices_df, data_timeframe)
            technical_analysis[ticker] = {
                "signal": intraday_result["signal"],
                "confidence": round(intraday_result["confidence"] * 100),
                "execution_window": intraday_result["execution_window"],
                "signal_ttl": intraday_result["signal_ttl"],
                "risk_parameters": intraday_result["risk_parameters"],
                "reasoning": normalize_pandas(intraday_result["reasoning"]),
            }
            progress.update_status(
                agent_id,
                ticker,
                "Done",
                analysis=json.dumps(technical_analysis[ticker], indent=4),
                context=progress_context,
            )
            continue

        # Daily swing/position pipeline
        prices = get_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
        )

        if not prices:
            progress.update_status(agent_id, ticker, "Failed: No price data found", context=progress_context)
            continue

        prices_df = prices_to_df(prices)

        progress.update_status(agent_id, ticker, "Calculating trend signals", context=progress_context)
        trend_signals = calculate_trend_signals(prices_df)

        progress.update_status(agent_id, ticker, "Calculating mean reversion", context=progress_context)
        mean_reversion_signals = calculate_mean_reversion_signals(prices_df)

        progress.update_status(agent_id, ticker, "Calculating momentum", context=progress_context)
        momentum_signals = calculate_momentum_signals(prices_df)

        progress.update_status(agent_id, ticker, "Analyzing volatility", context=progress_context)
        volatility_signals = calculate_volatility_signals(prices_df)

        progress.update_status(agent_id, ticker, "Statistical analysis", context=progress_context)
        stat_arb_signals = calculate_stat_arb_signals(prices_df)

        strategy_weights = {
            "trend": 0.25,
            "mean_reversion": 0.20,
            "momentum": 0.25,
            "volatility": 0.15,
            "stat_arb": 0.15,
        }

        progress.update_status(agent_id, ticker, "Combining signals", context=progress_context)
        combined_signal = weighted_signal_combination(
            {
                "trend": trend_signals,
                "mean_reversion": mean_reversion_signals,
                "momentum": momentum_signals,
                "volatility": volatility_signals,
                "stat_arb": stat_arb_signals,
            },
            strategy_weights,
        )

        atr_ratio = volatility_signals["metrics"].get("atr_ratio", 0.02)
        ttl_days = max(2, min(10, len(prices_df) // 15 or 3))
        execution_window = {
            "start": f"{start_date}T09:30:00",
            "end": f"{end_date}T16:00:00",
        }
        risk_parameters = {
            "stop_loss_pct": float(np.clip(safe_float(atr_ratio) * 3, 0.02, 0.12)),
            "take_profit_pct": float(np.clip(safe_float(atr_ratio) * 5, 0.04, 0.25)),
            "max_position_pct": 0.22,
            "volatility_target": float(volatility_signals["metrics"].get("historical_volatility", 0.25)),
        }

        technical_analysis[ticker] = {
            "signal": combined_signal["signal"],
            "confidence": round(combined_signal["confidence"] * 100),
            "execution_window": execution_window,
            "signal_ttl": f"{ttl_days}d",
            "risk_parameters": risk_parameters,
            "reasoning": {
                "trend_following": {
                    "signal": trend_signals["signal"],
                    "confidence": round(trend_signals["confidence"] * 100),
                    "metrics": normalize_pandas(trend_signals["metrics"]),
                },
                "mean_reversion": {
                    "signal": mean_reversion_signals["signal"],
                    "confidence": round(mean_reversion_signals["confidence"] * 100),
                    "metrics": normalize_pandas(mean_reversion_signals["metrics"]),
                },
                "momentum": {
                    "signal": momentum_signals["signal"],
                    "confidence": round(momentum_signals["confidence"] * 100),
                    "metrics": normalize_pandas(momentum_signals["metrics"]),
                },
                "volatility": {
                    "signal": volatility_signals["signal"],
                    "confidence": round(volatility_signals["confidence"] * 100),
                    "metrics": normalize_pandas(volatility_signals["metrics"]),
                },
                "statistical_arbitrage": {
                    "signal": stat_arb_signals["signal"],
                    "confidence": round(stat_arb_signals["confidence"] * 100),
                    "metrics": normalize_pandas(stat_arb_signals["metrics"]),
                },
                "ensemble": {
                    "signal": combined_signal["signal"],
                    "confidence": round(combined_signal["confidence"] * 100),
                    "score": safe_float(combined_signal.get("composite_score", 0.0)),
                },
            },
        }
        progress.update_status(
            agent_id,
            ticker,
            "Done",
            analysis=json.dumps(technical_analysis[ticker], indent=4),
            context=progress_context,
        )

    # Create the technical analyst message
    message = HumanMessage(
        content=json.dumps(technical_analysis),
        name=agent_id,
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(technical_analysis, "Technical Analyst")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = technical_analysis

    progress.update_status(agent_id, None, "Done", context=progress_context)

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }


def calculate_trend_signals(prices_df):
    """
    Advanced trend following strategy using multiple timeframes and indicators
    """
    # Calculate EMAs for multiple timeframes
    ema_8 = calculate_ema(prices_df, 8)
    ema_21 = calculate_ema(prices_df, 21)
    ema_55 = calculate_ema(prices_df, 55)

    # Calculate ADX for trend strength
    adx = calculate_adx(prices_df, 14)

    # Determine trend direction and strength
    short_trend = ema_8 > ema_21
    medium_trend = ema_21 > ema_55

    # Combine signals with confidence weighting
    trend_strength = adx["adx"].iloc[-1] / 100.0

    if short_trend.iloc[-1] and medium_trend.iloc[-1]:
        signal = "bullish"
        confidence = trend_strength
    elif not short_trend.iloc[-1] and not medium_trend.iloc[-1]:
        signal = "bearish"
        confidence = trend_strength
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "adx": safe_float(adx["adx"].iloc[-1]),
            "trend_strength": safe_float(trend_strength),
        },
    }


def calculate_mean_reversion_signals(prices_df):
    """
    Mean reversion strategy using statistical measures and Bollinger Bands
    """
    # Calculate z-score of price relative to moving average
    ma_50 = prices_df["close"].rolling(window=50).mean()
    std_50 = prices_df["close"].rolling(window=50).std()
    z_score = (prices_df["close"] - ma_50) / std_50

    # Calculate Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(prices_df)

    # Calculate RSI with multiple timeframes
    rsi_14 = calculate_rsi(prices_df, 14)
    rsi_28 = calculate_rsi(prices_df, 28)

    # Mean reversion signals
    price_vs_bb = (prices_df["close"].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

    # Combine signals
    if z_score.iloc[-1] < -2 and price_vs_bb < 0.2:
        signal = "bullish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    elif z_score.iloc[-1] > 2 and price_vs_bb > 0.8:
        signal = "bearish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "z_score": safe_float(z_score.iloc[-1]),
            "price_vs_bb": safe_float(price_vs_bb),
            "rsi_14": safe_float(rsi_14.iloc[-1]),
            "rsi_28": safe_float(rsi_28.iloc[-1]),
        },
    }


def calculate_momentum_signals(prices_df):
    """
    Multi-factor momentum strategy
    """
    # Price momentum
    returns = prices_df["close"].pct_change()
    mom_1m = returns.rolling(21).sum()
    mom_3m = returns.rolling(63).sum()
    mom_6m = returns.rolling(126).sum()

    # Volume momentum
    volume_ma = prices_df["volume"].rolling(21).mean()
    volume_momentum = prices_df["volume"] / volume_ma

    # Relative strength
    # (would compare to market/sector in real implementation)

    # Calculate momentum score
    momentum_score = (0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m).iloc[-1]

    # Volume confirmation
    volume_confirmation = volume_momentum.iloc[-1] > 1.0

    if momentum_score > 0.05 and volume_confirmation:
        signal = "bullish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    elif momentum_score < -0.05 and volume_confirmation:
        signal = "bearish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "momentum_1m": safe_float(mom_1m.iloc[-1]),
            "momentum_3m": safe_float(mom_3m.iloc[-1]),
            "momentum_6m": safe_float(mom_6m.iloc[-1]),
            "volume_momentum": safe_float(volume_momentum.iloc[-1]),
        },
    }


def calculate_volatility_signals(prices_df):
    """
    Volatility-based trading strategy
    """
    # Calculate various volatility metrics
    returns = prices_df["close"].pct_change()

    # Historical volatility
    hist_vol = returns.rolling(21).std() * math.sqrt(252)

    # Volatility regime detection
    vol_ma = hist_vol.rolling(63).mean()
    vol_regime = hist_vol / vol_ma

    # Volatility mean reversion
    vol_z_score = (hist_vol - vol_ma) / hist_vol.rolling(63).std()

    # ATR ratio
    atr = calculate_atr(prices_df)
    atr_ratio = atr / prices_df["close"]

    # Generate signal based on volatility regime
    current_vol_regime = vol_regime.iloc[-1]
    vol_z = vol_z_score.iloc[-1]

    if current_vol_regime < 0.8 and vol_z < -1:
        signal = "bullish"  # Low vol regime, potential for expansion
        confidence = min(abs(vol_z) / 3, 1.0)
    elif current_vol_regime > 1.2 and vol_z > 1:
        signal = "bearish"  # High vol regime, potential for contraction
        confidence = min(abs(vol_z) / 3, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "historical_volatility": safe_float(hist_vol.iloc[-1]),
            "volatility_regime": safe_float(current_vol_regime),
            "volatility_z_score": safe_float(vol_z),
            "atr_ratio": safe_float(atr_ratio.iloc[-1]),
        },
    }


def calculate_stat_arb_signals(prices_df):
    """
    Statistical arbitrage signals based on price action analysis
    """
    # Calculate price distribution statistics
    returns = prices_df["close"].pct_change()

    # Skewness and kurtosis
    skew = returns.rolling(63).skew()
    kurt = returns.rolling(63).kurt()

    # Test for mean reversion using Hurst exponent
    hurst = calculate_hurst_exponent(prices_df["close"])

    # Correlation analysis
    # (would include correlation with related securities in real implementation)

    # Generate signal based on statistical properties
    if hurst < 0.4 and skew.iloc[-1] > 1:
        signal = "bullish"
        confidence = (0.5 - hurst) * 2
    elif hurst < 0.4 and skew.iloc[-1] < -1:
        signal = "bearish"
        confidence = (0.5 - hurst) * 2
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "hurst_exponent": safe_float(hurst),
            "skewness": safe_float(skew.iloc[-1]),
            "kurtosis": safe_float(kurt.iloc[-1]),
        },
    }


def is_intraday_mode(strategy_mode: str | None, data_timeframe: str | None) -> bool:
    if strategy_mode and "intra" in strategy_mode.lower():
        return True
    if data_timeframe:
        normalized = data_timeframe.lower()
        return any(unit in normalized for unit in ["m", "min", "hour", "hr", "h"])
    return False


def timeframe_to_minutes(label: str | None, *, default: int = 30) -> int:
    if not label:
        return default
    match = re.match(r"(?i)(\d+)(?:\s*)([a-z]+)", label.strip())
    if not match:
        return default
    value = int(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("m"):
        return max(1, value)
    if unit.startswith("h"):
        return max(1, value * 60)
    if unit.startswith("d"):
        # Approximate US market hours (6.5 hours)
        return max(1, value * 390)
    return default


def analyze_intraday_signals(prices_df: pd.DataFrame, timeframe_label: str | None) -> dict:
    vwap = calculate_intraday_vwap(prices_df)
    crossover = detect_short_term_ma_crossover(prices_df)
    momentum = identify_momentum_bursts(prices_df)
    volume = detect_volume_anomalies(prices_df)

    components = {
        "vwap": vwap,
        "ma_crossover": crossover,
        "momentum": momentum,
        "volume_anomaly": volume,
    }

    score = float(sum(comp["score"] for comp in components.values()))
    normalized_score = score / max(len(components), 1)

    if normalized_score > 0.15:
        signal = "bullish"
    elif normalized_score < -0.15:
        signal = "bearish"
    else:
        signal = "neutral"

    confidence = min(1.0, max(0.05, abs(normalized_score)))
    timeframe_minutes = timeframe_to_minutes(timeframe_label, default=30)
    ttl_minutes = max(15, timeframe_minutes * 2)

    last_timestamp = pd.to_datetime(prices_df["time"].iloc[-1])
    window = pd.Timedelta(minutes=ttl_minutes)
    execution_window = {
        "start": (last_timestamp - window).isoformat(),
        "end": (last_timestamp + window).isoformat(),
    }

    risk_parameters = build_intraday_risk_parameters(prices_df, timeframe_minutes)

    reasoning = {
        name: {
            "signal": comp["signal"],
            "score": round(safe_float(comp["score"]), 3),
            "metrics": normalize_pandas(comp["metrics"]),
        }
        for name, comp in components.items()
    }
    reasoning["composite"] = {
        "signal": signal,
        "score": round(normalized_score, 3),
    }

    return {
        "signal": signal,
        "confidence": confidence,
        "execution_window": execution_window,
        "signal_ttl": f"{ttl_minutes}m",
        "risk_parameters": risk_parameters,
        "reasoning": reasoning,
    }


def calculate_intraday_vwap(prices_df: pd.DataFrame) -> dict:
    df = prices_df.copy()
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["cum_tp_vol"] = (df["typical_price"] * df["volume"]).cumsum()
    df["cum_vol"] = df["volume"].cumsum().replace(0, np.nan)
    df["vwap"] = df["cum_tp_vol"] / df["cum_vol"]

    latest = df.iloc[-1]
    vwap_value = safe_float(latest.get("vwap"), default=latest["close"])
    deviation = 0.0
    if vwap_value:
        deviation = safe_float((latest["close"] - vwap_value) / vwap_value)

    if deviation > 0.002:
        signal = "bullish"
    elif deviation < -0.002:
        signal = "bearish"
    else:
        signal = "neutral"

    score = float(np.clip(deviation * 10, -1.0, 1.0))
    metrics = {
        "vwap": vwap_value,
        "price_vs_vwap": deviation,
    }
    return {"signal": signal, "score": score, "metrics": metrics}


def detect_short_term_ma_crossover(prices_df: pd.DataFrame) -> dict:
    close = prices_df["close"]
    fast = close.rolling(window=5).mean()
    slow = close.rolling(window=21).mean()

    if len(fast.dropna()) == 0 or len(slow.dropna()) == 0:
        return {"signal": "neutral", "score": 0.0, "metrics": {"fast_ma": None, "slow_ma": None}}

    diff = safe_float(fast.iloc[-1] - slow.iloc[-1])
    slope = safe_float(fast.diff().iloc[-1])

    if diff > 0 and slope > 0:
        signal = "bullish"
    elif diff < 0 and slope < 0:
        signal = "bearish"
    else:
        signal = "neutral"

    score = float(np.clip((diff / slow.iloc[-1]) * 5 if slow.iloc[-1] else 0, -1.0, 1.0))
    metrics = {
        "fast_ma": safe_float(fast.iloc[-1]),
        "slow_ma": safe_float(slow.iloc[-1]),
        "ma_slope": slope,
    }
    return {"signal": signal, "score": score, "metrics": metrics}


def identify_momentum_bursts(prices_df: pd.DataFrame) -> dict:
    returns = prices_df["close"].pct_change().dropna()
    if returns.empty:
        return {"signal": "neutral", "score": 0.0, "metrics": {}}

    short_window = returns.tail(5)
    burst = safe_float(short_window.sum())
    volatility = safe_float(returns.tail(30).std()) or 1e-4
    burst_score = float(np.clip(burst / (volatility * 3), -1.5, 1.5))

    if burst_score > 0.2:
        signal = "bullish"
    elif burst_score < -0.2:
        signal = "bearish"
    else:
        signal = "neutral"

    metrics = {
        "short_term_return": burst,
        "volatility": volatility,
    }
    return {"signal": signal, "score": burst_score, "metrics": metrics}


def detect_volume_anomalies(prices_df: pd.DataFrame) -> dict:
    volume = prices_df["volume"].fillna(0)
    if volume.empty:
        return {"signal": "neutral", "score": 0.0, "metrics": {}}

    avg_volume = volume.rolling(window=30, min_periods=5).mean().iloc[-1]
    latest_volume = safe_float(volume.iloc[-1])
    if not avg_volume or avg_volume == 0:
        ratio = 1.0
    else:
        ratio = latest_volume / avg_volume

    if ratio > 1.5:
        signal = "bullish"
    elif ratio < 0.7:
        signal = "bearish"
    else:
        signal = "neutral"

    score = float(np.clip((ratio - 1.0), -1.0, 1.0))
    metrics = {
        "volume_ratio": ratio,
        "average_volume": avg_volume,
        "latest_volume": latest_volume,
    }
    return {"signal": signal, "score": score, "metrics": metrics}


def build_intraday_risk_parameters(prices_df: pd.DataFrame, timeframe_minutes: int) -> dict:
    returns = prices_df["close"].pct_change().dropna()
    if returns.empty:
        return {
            "stop_loss_pct": 0.01,
            "take_profit_pct": 0.02,
            "max_position_pct": 0.12,
            "volatility_target": 0.20,
        }

    window = max(5, min(len(returns), timeframe_minutes))
    recent = returns.tail(window)
    realized_vol = float(recent.std())
    stop_loss = float(np.clip(realized_vol * 4, 0.005, 0.06))
    take_profit = float(np.clip(realized_vol * 6, 0.01, 0.10))
    max_position = 0.10 if timeframe_minutes <= 30 else 0.15

    return {
        "stop_loss_pct": stop_loss,
        "take_profit_pct": take_profit,
        "max_position_pct": max_position,
        "volatility_target": float(np.sqrt(252) * realized_vol),
    }


def weighted_signal_combination(signals, weights):
    """
    Combines multiple trading signals using a weighted approach
    """
    # Convert signals to numeric values
    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}

    weighted_sum = 0
    total_confidence = 0

    for strategy, signal in signals.items():
        numeric_signal = signal_values[signal["signal"]]
        weight = weights[strategy]
        confidence = signal["confidence"]

        weighted_sum += numeric_signal * weight * confidence
        total_confidence += weight * confidence

    # Normalize the weighted sum
    if total_confidence > 0:
        final_score = weighted_sum / total_confidence
    else:
        final_score = 0

    # Convert back to signal
    if final_score > 0.2:
        signal = "bullish"
    elif final_score < -0.2:
        signal = "bearish"
    else:
        signal = "neutral"

    return {"signal": signal, "confidence": min(1.0, abs(final_score)), "composite_score": final_score}


def normalize_pandas(obj):
    """Convert pandas Series/DataFrames to primitive Python types"""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    elif isinstance(obj, dict):
        return {k: normalize_pandas(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_pandas(item) for item in obj]
    return obj


def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = prices_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(prices_df: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average

    Args:
        df: DataFrame with price data
        window: EMA period

    Returns:
        pd.Series: EMA values
    """
    return df["close"].ewm(span=window, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX)

    Args:
        df: DataFrame with OHLC data
        period: Period for calculations

    Returns:
        DataFrame with ADX values
    """
    # Calculate True Range
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)

    # Calculate Directional Movement
    df["up_move"] = df["high"] - df["high"].shift()
    df["down_move"] = df["low"].shift() - df["low"]

    df["plus_dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
    df["minus_dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)

    # Calculate ADX
    df["+di"] = 100 * (df["plus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["-di"] = 100 * (df["minus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["dx"] = 100 * abs(df["+di"] - df["-di"]) / (df["+di"] + df["-di"])
    df["adx"] = df["dx"].ewm(span=period).mean()

    return df[["adx", "+di", "-di"]]


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range

    Args:
        df: DataFrame with OHLC data
        period: Period for ATR calculation

    Returns:
        pd.Series: ATR values
    """
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    return true_range.rolling(period).mean()


def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate Hurst Exponent to determine long-term memory of time series
    H < 0.5: Mean reverting series
    H = 0.5: Random walk
    H > 0.5: Trending series

    Args:
        price_series: Array-like price data
        max_lag: Maximum lag for R/S calculation

    Returns:
        float: Hurst exponent
    """
    lags = range(2, max_lag)
    # Add small epsilon to avoid log(0)
    tau = [max(1e-8, np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag])))) for lag in lags]

    # Return the Hurst exponent from linear fit
    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]  # Hurst exponent is the slope
    except (ValueError, RuntimeWarning):
        # Return 0.5 (random walk) if calculation fails
        return 0.5
