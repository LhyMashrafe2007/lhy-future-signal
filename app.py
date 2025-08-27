# app.py
import os
import time
from flask import Flask, request, jsonify, render_template, make_response
import requests

# ====== Configuration ======
app = Flask(__name__, template_folder="templates")

# Prefer environment variable for security. Fallback provided (you can remove fallback).
API_KEY = os.environ.get("TD_API_KEY", "34d108698a5044abb24fbdee2ef0cd91")

# Simple in-memory cache to avoid hitting rate limits: { symbol_upper: { 'ts': epoch, 'vals': [...] } }
CACHE = {}
CACHE_TTL = 30  # seconds

# ====== Utilities ======
def json_response(payload, status=200):
    """Always return JSON (so frontend never gets HTML). Also add CORS headers for safety."""
    resp = make_response(jsonify(payload), status)
    resp.headers["Content-Type"] = "application/json"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

@app.after_request
def add_cors_headers(response):
    # Ensure CORS for all responses (harmless and useful for debugging)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# ====== Data fetching with cache ======
def fetch_data_from_twelvedata(symbol):
    """
    Fetch 1-min OHLC candles from Twelve Data for given symbol (e.g. "EUR/USD").
    Returns list of candles oldest-first (each item: dict with 'datetime','open','high','low','close').
    On error returns [].
    """
    key = str(symbol).upper()
    now = time.time()
    # return cached
    if key in CACHE and (now - CACHE[key]['ts'] < CACHE_TTL):
        return CACHE[key]['vals']

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1min",
        "outputsize": 200,
        "apikey": API_KEY
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        data = r.json()
    except Exception as e:
        app.logger.error(f"Error fetching {symbol}: {e}")
        return []

    # handle API errors gracefully
    if not isinstance(data, dict):
        app.logger.error(f"TwelveData returned non-dict for {symbol}: {data}")
        return []

    # If Twelve Data returns error message
    if data.get("status") == "error" or "values" not in data:
        app.logger.error(f"TwelveData error for {symbol}: {data}")
        return []

    vals = data.get("values", []) or []
    # API returns newest-first; we want oldest-first for indicator calculations
    vals = list(reversed(vals))
    CACHE[key] = {'ts': now, 'vals': vals}
    return vals

# ====== Indicator calculations (pure Python) ======
def compute_indicators(closes, highs, lows):
    """
    Given lists of closes/highs/lows (oldest-first), compute:
      - ema20 list
      - ema70 list
      - rsi14 list
      - atr14 list
    Each list is same length; indices with insufficient history -> None.
    """
    n = len(closes)
    ema20 = [None] * n
    ema70 = [None] * n
    rsi = [None] * n
    atr = [None] * n

    # EMA20
    if n >= 20:
        sma20 = sum(closes[0:20]) / 20.0
        ema20[19] = sma20
        alpha20 = 2.0 / (20 + 1)
        for i in range(20, n):
            ema20[i] = closes[i] * alpha20 + ema20[i-1] * (1 - alpha20)

    # EMA70
    if n >= 70:
        sma70 = sum(closes[0:70]) / 70.0
        ema70[69] = sma70
        alpha70 = 2.0 / (70 + 1)
        for i in range(70, n):
            ema70[i] = closes[i] * alpha70 + ema70[i-1] * (1 - alpha70)

    # RSI14 (Wilder)
    if n >= 15:
        gains = 0.0
        losses = 0.0
        # first 14 periods' average gain/loss
        for i in range(1, 15):
            delta = closes[i] - closes[i-1]
            if delta > 0:
                gains += delta
            else:
                losses -= delta
        avg_gain = gains / 14.0
        avg_loss = losses / 14.0
        if avg_loss == 0:
            rsi[14] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[14] = 100.0 - (100.0 / (1 + rs))
        for i in range(15, n):
            delta = closes[i] - closes[i-1]
            gain = delta if delta > 0 else 0.0
            loss = -delta if delta < 0 else 0.0
            avg_gain = (avg_gain * 13 + gain) / 14.0
            avg_loss = (avg_loss * 13 + loss) / 14.0
            if avg_loss == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - (100.0 / (1 + rs))

    # ATR14
    if n >= 15:
        tr_sum = 0.0
        for i in range(1, 15):
            prev = closes[i-1]
            tr = max(highs[i] - lows[i], abs(highs[i] - prev), abs(prev - lows[i]))
            tr_sum += tr
        atr[14] = tr_sum / 14.0
        for i in range(15, n):
            prev = closes[i-1]
            tr = max(highs[i] - lows[i], abs(highs[i] - prev), abs(prev - lows[i]))
            atr[i] = (atr[i-1] * 13 + tr) / 14.0

    return ema20, ema70, rsi, atr

# ====== Signal detection ======
def detect_signals_for_symbol(symbol, candles, direction_filter="BOTH", start_time=None, end_time=None):
    """
    scans candles and returns list of signals (newest-first).
    direction_filter: "BOTH", "CALL" (BUY), "PUT" (SELL)
    start_time / end_time: optional "HH:MM" filters (local string from candle datetime)
    """
    if not candles or len(candles) < 75:
        return []

    n = len(candles)
    closes = [float(c['close']) for c in candles]
    highs = [float(c['high']) for c in candles]
    lows = [float(c['low']) for c in candles]
    times = [c.get('datetime') or c.get('timestamp') for c in candles]

    ema20, ema70, rsi, atr = compute_indicators(closes, highs, lows)
    signals = []

    # start from index 70 (where ema70 becomes available)
    for i in range(70, n):
        if ema20[i-1] is None or ema70[i-1] is None or ema20[i] is None or ema70[i] is None:
            continue

        # optional time-window filter
        if start_time or end_time:
            hhmm = ""
            try:
                hhmm = str(times[i]).split(" ")[1][:5]
            except Exception:
                hhmm = ""
            ok = True
            if start_time and hhmm:
                ok = ok and (hhmm >= start_time)
            if end_time and hhmm:
                ok = ok and (hhmm <= end_time)
            if not ok:
                continue

        # Bullish crossover: ema20 crosses above ema70
        if ema20[i-1] < ema70[i-1] and ema20[i] > ema70[i]:
            # require RSI confirmation: oversold (reversal)
            if rsi[i] is not None and rsi[i] < 30:
                if direction_filter in ("BOTH", "CALL"):
                    entry = closes[i]
                    atr_v = atr[i] or 0.0
                    sl = entry - atr_v
                    tp = entry + 2.0 * atr_v
                    signals.append({
                        "time": times[i],
                        "pair": symbol,
                        "type": "BUY",
                        "price": round(entry, 6),
                        "sl": round(sl, 6),
                        "tp": round(tp, 6),
                        "reason": "20 EMA crossed above 70 EMA + RSI oversold (<30)"
                    })

        # Bearish crossover: ema20 crosses below ema70
        if ema20[i-1] > ema70[i-1] and ema20[i] < ema70[i]:
            # require RSI confirmation: overbought
            if rsi[i] is not None and rsi[i] > 70:
                if direction_filter in ("BOTH", "PUT"):
                    entry = closes[i]
                    atr_v = atr[i] or 0.0
                    sl = entry + atr_v
                    tp = entry - 2.0 * atr_v
                    signals.append({
                        "time": times[i],
                        "pair": symbol,
                        "type": "SELL",
                        "price": round(entry, 6),
                        "sl": round(sl, 6),
                        "tp": round(tp, 6),
                        "reason": "20 EMA crossed below 70 EMA + RSI overbought (>70)"
                    })

    # newest-first
    signals_sorted = sorted(signals, key=lambda x: x['time'], reverse=True)
    return signals_sorted

# ====== Routes ======
@app.route("/")
def index():
    # Render templates/index.html (you told index.html is inside templates/)
    return render_template("index.html")

@app.route("/signals", methods=["POST", "OPTIONS"])
def signals_endpoint():
    # Handle preflight OPTIONS quickly
    if request.method == "OPTIONS":
        return json_response({"ok": True})

    # Validate JSON
    if not request.is_json:
        return json_response({"error": "Request must be application/json"}, status=400)
    body = request.get_json() or {}

    pairs = body.get("pairs")
    if not pairs or not isinstance(pairs, list):
        return json_response({"error": "Provide 'pairs' as a list in JSON body"}, status=400)

    # Acceptable direction values from frontend: BOTH | CALL | PUT
    direction = body.get("signalDirection", "BOTH")
    if direction not in ("BOTH", "CALL", "PUT"):
        direction = "BOTH"

    start_time = body.get("startTime")  # "HH:MM" optional
    end_time = body.get("endTime")
    try:
        limit = int(body.get("limit", 10))
    except Exception:
        limit = 10
    # enforce limit boundaries
    if limit <= 0:
        limit = 10
    if limit > 10:
        limit = 10

    all_signals = []
    for p in pairs:
        symbol = str(p).strip().upper()
        candles = fetch_data_from_twelvedata(symbol)
        if not candles:
            # include debug info but don't return HTML
            app.logger.warning(f"No candle data for {symbol}")
            continue
        sigs = detect_signals_for_symbol(symbol, candles, direction_filter=direction, start_time=start_time, end_time=end_time)
        if sigs:
            all_signals.extend(sigs)

    # sort all signals by time (newest first) and limit
    all_signals.sort(key=lambda x: x['time'], reverse=True)
    limited = all_signals[:limit]

    return json_response({"signals": limited})

# ====== Error handlers to ensure JSON responses ======
@app.errorhandler(404)
def not_found(e):
    return json_response({"error": "Not Found"}, status=404)

@app.errorhandler(500)
def server_error(e):
    app.logger.exception("Server error")
    return json_response({"error": "Internal Server Error"}, status=500)

# ====== Run ======
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
