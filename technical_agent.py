import pandas as pd
import yfinance as yf
from typing import Any, TypedDict
from langgraph.graph import StateGraph, START, END

import llm


class technical_analysis_agent(TypedDict):
    context: Any

    # raw history is stored as a plain, JSON/checkpoint-safe dict:
    # {'index': [...], 'columns': [...], 'data': [[...], ...]}
    # -> reconstructed into a pandas DataFrame only inside the nodes
    #    that actually need it.
    history: Any

    moving_average: Any
    rsi: Any
    macd: Any
    bollinger_bands: Any
    volume_profile: Any
    trend: Any
    support_resistance: Any

    analysis: Any  # final LLM output


# ---------------------------------------------------------------------
# small local helpers (plain-python, no pandas/numpy needed once the
# columns have been pulled out of the DataFrame)
# ---------------------------------------------------------------------

def _sma(series, window):
    out = []
    for i in range(len(series)):
        if i < window - 1:
            out.append(None)
        else:
            out.append(sum(series[i - window + 1:i + 1]) / window)
    return out


def _ema(series, window):
    k = 2 / (window + 1)
    out = [series[0]]
    for price in series[1:]:
        out.append(price * k + out[-1] * (1 - k))
    return out


def _stdev(series, window):
    out = []
    for i in range(len(series)):
        if i < window - 1:
            out.append(None)
            continue
        chunk = series[i - window + 1:i + 1]
        mean = sum(chunk) / window
        variance = sum((x - mean) ** 2 for x in chunk) / window
        out.append(variance ** 0.5)
    return out


# ---------------------------------------------------------------------
# history <-> dict helpers
# ---------------------------------------------------------------------

def _df_to_state_dict(df: pd.DataFrame) -> dict:
    """
    Convert a price-history DataFrame into a plain python dict so it can
    live safely inside LangGraph state / checkpoints (config-serializable,
    no numpy/pandas objects).
    """
    return {
        "index": df.index.astype(str).tolist(),
        "columns": df.columns.tolist(),
        "data": df.values.tolist(),  # .tolist() casts numpy -> native python
    }


def _state_dict_to_df(history: dict) -> pd.DataFrame:
    """
    Rebuild the DataFrame from the dict stored in state. Only called by
    nodes that actually need DataFrame-style access.
    """
    df = pd.DataFrame(
        data=history["data"],
        index=pd.to_datetime(history["index"]),
        columns=history["columns"],
    )
    return df


def _get_history_df(state: technical_analysis_agent) -> pd.DataFrame:
    if not state.get("history"):
        raise ValueError(
            "state['history'] is empty - fetch_history must run before "
            "any indicator node."
        )
    return _state_dict_to_df(state["history"])


# ---------------------------------------------------------------------
# fetch_history: the ONLY node that hits the network. Runs exactly once,
# stores the raw OHLCV history into state as a dict.
# ---------------------------------------------------------------------

def fetch_history(state: technical_analysis_agent):
    ctx = state["context"]
    ticker_symbol = ctx["ticker"]

    # allow the caller to override via context, otherwise sane defaults
    period = ctx.get("period", "1y")
    interval = ctx.get("interval", "1d")

    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(
            f"No price history returned for '{ticker_symbol}' "
            f"(period={period}, interval={interval})"
        )

    return {"history": _df_to_state_dict(df)}


# ---------------------------------------------------------------------
# Moving Average Analysis
# ---------------------------------------------------------------------

def moving_average_analysis(state: technical_analysis_agent):
    df = _get_history_df(state)
    close = df["Close"].tolist()

    sma_short = _sma(close, 20)
    sma_long = _sma(close, 50)

    crossover = None
    if sma_short[-2] is not None and sma_long[-2] is not None:
        prev_diff = sma_short[-2] - sma_long[-2]
        curr_diff = sma_short[-1] - sma_long[-1]
        if prev_diff < 0 and curr_diff > 0:
            crossover = "golden_cross"
        elif prev_diff > 0 and curr_diff < 0:
            crossover = "death_cross"

    trend_bias = "bullish" if sma_short[-1] > sma_long[-1] else "bearish"

    result = {
        "sma_short": sma_short,
        "sma_long": sma_long,
        "trend_bias": trend_bias,
        "crossover_signal": crossover,
    }

    return {"moving_average": result}


# ---------------------------------------------------------------------
# RSI Analysis
# ---------------------------------------------------------------------

def rsi_analysis(state: technical_analysis_agent):
    df = _get_history_df(state)
    close = df["Close"].tolist()
    period = 14

    gains = [max(c2 - c1, 0) for c1, c2 in zip(close[:-1], close[1:])]
    losses = [max(c1 - c2, 0) for c1, c2 in zip(close[:-1], close[1:])]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    rsi_values = []
    for gain, loss in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
        rsi_values.append(100 - (100 / (1 + rs)))

    latest_rsi = round(rsi_values[-1], 2)
    if latest_rsi >= 70:
        condition = "overbought"
    elif latest_rsi <= 30:
        condition = "oversold"
    else:
        condition = "neutral"

    result = {
        "rsi_values": rsi_values,
        "latest_rsi": latest_rsi,
        "condition": condition,
    }

    return {"rsi": result}


# ---------------------------------------------------------------------
# MACD Analysis
# ---------------------------------------------------------------------

def macd_analysis(state: technical_analysis_agent):
    df = _get_history_df(state)
    close = df["Close"].tolist()

    ema_fast = _ema(close, 12)
    ema_slow = _ema(close, 26)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = _ema(macd_line, 9)
    histogram = [m - s for m, s in zip(macd_line, signal_line)]

    crossover = None
    prev_hist, curr_hist = histogram[-2], histogram[-1]
    if prev_hist < 0 and curr_hist > 0:
        crossover = "bullish_crossover"
    elif prev_hist > 0 and curr_hist < 0:
        crossover = "bearish_crossover"

    result = {
        "macd_line": macd_line,
        "signal_line": signal_line,
        "histogram": histogram,
        "latest_histogram": round(curr_hist, 3),
        "crossover_signal": crossover,
    }

    return {"macd": result}


# ---------------------------------------------------------------------
# Bollinger Band Analysis
# ---------------------------------------------------------------------

def bollinger_band_analysis(state: technical_analysis_agent):
    df = _get_history_df(state)
    close = df["Close"].tolist()
    window = 20

    mid_band = _sma(close, window)
    std = _stdev(close, window)

    upper_band = [m + 2 * s if m is not None else None for m, s in zip(mid_band, std)]
    lower_band = [m - 2 * s if m is not None else None for m, s in zip(mid_band, std)]

    latest_price = close[-1]
    latest_upper = upper_band[-1]
    latest_lower = lower_band[-1]

    if latest_price >= latest_upper:
        position = "above_upper_band"
    elif latest_price <= latest_lower:
        position = "below_lower_band"
    else:
        position = "within_bands"

    result = {
        "mid_band": mid_band,
        "upper_band": upper_band,
        "lower_band": lower_band,
        "latest_price": latest_price,
        "position": position,
    }

    return {"bollinger_bands": result}


# ---------------------------------------------------------------------
# Volume Analysis
# ---------------------------------------------------------------------

def volume_analysis(state: technical_analysis_agent):
    df = _get_history_df(state)
    close = df["Close"].tolist()
    volume = df["Volume"].tolist()
    window = 20

    avg_volume = _sma(volume, window)
    latest_volume = volume[-1]
    latest_avg_volume = avg_volume[-1]
    volume_ratio = round(latest_volume / latest_avg_volume, 2) if latest_avg_volume else None

    directions = [1 if c2 > c1 else (-1 if c2 < c1 else 0) for c1, c2 in zip(close[:-1], close[1:])]
    obv = [0]
    for direction, vol in zip(directions, volume[1:]):
        obv.append(obv[-1] + direction * vol)

    if volume_ratio is not None and volume_ratio >= 1.5:
        volume_signal = "unusually_high_volume"
    elif volume_ratio is not None and volume_ratio <= 0.5:
        volume_signal = "unusually_low_volume"
    else:
        volume_signal = "normal_volume"

    result = {
        "avg_volume": avg_volume,
        "obv": obv,
        "latest_volume": latest_volume,
        "volume_ratio": volume_ratio,
        "volume_signal": volume_signal,
    }

    return {"volume_profile": result}


# ---------------------------------------------------------------------
# Trend Analysis (uses only the most recent 30-day window of the
# already-fetched history - no separate fetch needed)
# ---------------------------------------------------------------------

def trend_analysis(state: technical_analysis_agent):
    df = _get_history_df(state)
    window = 30
    close = df["Close"].tolist()[-window:]

    n = len(close)
    x_mean = (n - 1) / 2
    y_mean = sum(close) / n

    numerator = sum((i - x_mean) * (c - y_mean) for i, c in enumerate(close))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    slope = numerator / denominator if denominator != 0 else 0

    pct_slope_per_day = slope / y_mean * 100

    if pct_slope_per_day > 0.05:
        direction = "uptrend"
    elif pct_slope_per_day < -0.05:
        direction = "downtrend"
    else:
        direction = "sideways"

    result = {
        "slope_pct_per_day": round(pct_slope_per_day, 4),
        "direction": direction,
    }

    return {"trend": result}


# ---------------------------------------------------------------------
# Support / Resistance
# ---------------------------------------------------------------------

def support_resistance(state: technical_analysis_agent):
    df = _get_history_df(state)
    high = df["High"].tolist()
    low = df["Low"].tolist()
    close = df["Close"].tolist()
    window = 5

    resistance_points = []
    support_points = []

    for i in range(window, len(close) - window):
        local_high = high[i - window:i + window + 1]
        local_low = low[i - window:i + window + 1]

        if high[i] == max(local_high):
            resistance_points.append(high[i])
        if low[i] == min(local_low):
            support_points.append(low[i])

    latest_price = close[-1]
    resistances_above = sorted([r for r in resistance_points if r > latest_price])[:3]
    supports_below = sorted([s for s in support_points if s < latest_price], reverse=True)[:3]

    result = {
        "nearest_resistances": resistances_above,
        "nearest_supports": supports_below,
    }

    return {"support_resistance": result}


# ---------------------------------------------------------------------
# Final Technical Summary (LLM synthesis)
# ---------------------------------------------------------------------

def _summarize_for_llm(state: technical_analysis_agent) -> dict:
    """
    Pull out only the latest scalar readings / signals from each
    indicator's result. The full time-series arrays (sma_short, rsi_values,
    macd_line, histogram, upper/lower bands, obv, ...) stay in state for
    charting/debugging but are never sent to the LLM - they're irrelevant
    to the summary and just burn tokens.
    """
    ma = state["moving_average"]
    rsi = state["rsi"]
    macd = state["macd"]
    bb = state["bollinger_bands"]
    vol = state["volume_profile"]
    trend = state["trend"]
    sr = state["support_resistance"]

    return {
        "moving_average": {
            "sma_20": round(ma["sma_short"][-1], 2),
            "sma_50": round(ma["sma_long"][-1], 2),
            "trend_bias": ma["trend_bias"],
            "crossover_signal": ma["crossover_signal"],
        },
        "rsi": {
            "latest_rsi": rsi["latest_rsi"],
            "condition": rsi["condition"],
        },
        "macd": {
            "latest_histogram": macd["latest_histogram"],
            "crossover_signal": macd["crossover_signal"],
        },
        "bollinger_bands": {
            "latest_price": round(bb["latest_price"], 2),
            "upper_band": round(bb["upper_band"][-1], 2),
            "lower_band": round(bb["lower_band"][-1], 2),
            "position": bb["position"],
        },
        "volume_profile": {
            "latest_volume": vol["latest_volume"],
            "volume_ratio": vol["volume_ratio"],
            "volume_signal": vol["volume_signal"],
            "obv_latest": vol["obv"][-1],
        },
        "trend": trend,
        "support_resistance": sr,
    }


def technical_summary(state: technical_analysis_agent):
    summary = _summarize_for_llm(state)

    prompt = f"""
You are a technical analyst. Analyze the following indicators for {state['context']['ticker']}:

Moving Averages: {summary['moving_average']}
RSI: {summary['rsi']}
MACD: {summary['macd']}
Bollinger Bands: {summary['bollinger_bands']}
Volume Profile: {summary['volume_profile']}
Trend: {summary['trend']}
Support/Resistance: {summary['support_resistance']}

Based on this data:
1. State the overall trend bias (bullish, bearish, or neutral) with reasoning.
2. Highlight any conflicting signals between indicators.
3. Call out the nearest support and resistance levels and what a break of either would imply.
4. Give an overall verdict on the stock's technical setup (Strong Buy / Buy / Hold / Sell / Strong Sell), with reasoning.

Respond in a structured format with clear headers for each section.
"""
    response = llm.llm_normal.invoke(prompt)
    return {"analysis": response.content}


# ---------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------

builder = StateGraph(technical_analysis_agent)

# Single network call - executed exactly once
builder.add_node("fetch_history", fetch_history)

# Indicator nodes - all independent, run in parallel off the shared history
builder.add_node("moving_average_analysis", moving_average_analysis)
builder.add_node("rsi_analysis", rsi_analysis)
builder.add_node("macd_analysis", macd_analysis)
builder.add_node("bollinger_band_analysis", bollinger_band_analysis)
builder.add_node("volume_analysis", volume_analysis)
builder.add_node("trend_analysis", trend_analysis)
builder.add_node("support_resistance", support_resistance)

# Final synthesis node
builder.add_node("technical_summary", technical_summary)

builder.add_edge(START, "fetch_history")

# Fan-out: fetch_history -> all indicator nodes (parallel)
for indicator_node in [
    "moving_average_analysis", "rsi_analysis", "macd_analysis",
    "bollinger_band_analysis", "volume_analysis", "trend_analysis",
    "support_resistance",
]:
    builder.add_edge("fetch_history", indicator_node)

# Everything feeds into the final summary node
for node in [
    "moving_average_analysis", "rsi_analysis", "macd_analysis",
    "bollinger_band_analysis", "volume_analysis", "trend_analysis",
    "support_resistance",
]:
    builder.add_edge(node, "technical_summary")

builder.add_edge("technical_summary", END)

graph = builder.compile()