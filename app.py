# app.py
import os
import time
from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__, template_folder="templates")

# Use env var if available (set TD_API_KEY in Render); fallback to provided key
API_KEY = os.environ.get("TD_API_KEY", "34d108698a5044abb24fbdee2ef0cd91")
CACHE = {}
CACHE_TTL = 30  # seconds

def fetch_data_from_twelvedata(symbol):
    now = time.time()
    key = str(symbol).upper()
    if key in CACHE and now - CACHE[key]['ts'] < CACHE_TTL:
        return CACHE[key]['vals']

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1min",
        "outputsize": 200,
        "apikey": API_KEY
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
    except Exception as e:
        print("Fetch error for", symbol, "->", e)
        return []

    if not isinstance(data, dict) or "values" not in data:
        print("Twelve Data error or missing values for", symbol, "->", data)
        return []

    vals = data.get("values", [])
    vals = list(reversed(vals))  # oldest first
    CACHE[key] = {'ts': now, 'vals': vals}
    return vals

def compute_indicators(closes, highs, lows):
    n = len(closes)
    ema20 = [None]*n
    ema70 = [None]*n
    rsi = [None]*n
    atr = [None]*n

    # EMA20
    if n >= 20:
        sma20 = sum(closes[0:20]) / 20.0
        ema20[19] = sma20
        alpha20 = 2.0 / (20 + 1)
        for i in range(20, n):
            ema20[i] = closes[i]*alpha20 + ema20[i-1]*(1-alpha20)

    # EMA70
    if n >= 70:
        sma70 = sum(closes[0:70]) / 70.0
        ema70[69] = sma70
        alpha70 = 2.0 / (70 + 1)
        for i in range(70, n):
            ema70[i] = closes[i]*alpha70 + ema70[i-1]*(1-alpha70)

    # RSI14 (Wilder)
    if n >= 15:
        gains = 0.0
        losses = 0.0
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

def detect_signals(symbol, candles, direction_filter="BOTH", start_time=None, end_time=None):
    if not candles or len(candles) < 75:
        return []

    n = len(candles)
    closes = [float(c['close']) for c in candles]
    highs  = [float(c['high']) for c in candles]
    lows   = [float(c['low']) for c in candles]
    times  = [c.get('datetime') or c.get('timestamp') for c in candles]

    ema20, ema70, rsi, atr = compute_indicators(closes, highs, lows)
    sigs = []
    for i in range(70, n):
        if ema20[i-1] is None or ema70[i-1] is None or ema20[i] is None or ema70[i] is None:
            continue

        # time filter (optional)
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

        # bullish crossover
        if ema20[i-1] < ema70[i-1] and ema20[i] > ema70[i]:
            if rsi[i] is not None and rsi[i] < 30:
                if direction_filter in ("BOTH", "CALL"):
                    entry = closes[i]
                    atr_v = atr[i] or 0.0
                    sl = entry - atr_v
                    tp = entry + 2.0 * atr_v
                    sigs.append({
                        "time": times[i],
                        "pair": symbol,
                        "type": "BUY",
                        "price": round(entry, 6),
                        "sl": round(sl, 6),
                        "tp": round(tp, 6),
                        "reason": "20 EMA ↑ 70 EMA + RSI oversold (<30)"
                    })

        # bearish crossover
        if ema20[i-1] > ema70[i-1] and ema20[i] < ema70[i]:
            if rsi[i] is not None and rsi[i] > 70:
                if direction_filter in ("BOTH", "PUT"):
                    entry = closes[i]
                    atr_v = atr[i] or 0.0
                    sl = entry + atr_v
                    tp = entry - 2.0 * atr_v
                    sigs.append({
                        "time": times[i],
                        "pair": symbol,
                        "type": "SELL",
                        "price": round(entry, 6),
                        "sl": round(sl, 6),
                        "tp": round(tp, 6),
                        "reason": "20 EMA ↓ 70 EMA + RSI overbought (>70)"
                    })

    return sorted(sigs, key=lambda x: x['time'], reverse=True)

@app.route('/')
def index():
    # render templates/index.html (ensure file exists at templates/index.html)
    return render_template('index.html')

@app.route('/signals', methods=['POST'])
def signals_endpoint():
    if not request.is_json:
        return jsonify({"error": "Request must be application/json"}), 400

    body = request.get_json()
    pairs = body.get('pairs')
    if not pairs or not isinstance(pairs, list):
        return jsonify({"error": "Provide 'pairs' as a list in JSON body"}), 400

    direction = body.get('signalDirection', 'BOTH')  # BOTH|CALL|PUT
    start_time = body.get('startTime')
    end_time = body.get('endTime')
    try:
        limit = int(body.get('limit', 10))
    except Exception:
        limit = 10
    if limit <= 0:
        limit = 10
    if limit > 10:
        limit = 10

    all_signals = []
    for p in pairs:
        symbol = str(p).strip().upper()
        candles = fetch_data_from_twelvedata(symbol)
        if not candles:
            continue
        sigs = detect_signals(symbol, candles, direction_filter=direction, start_time=start_time, end_time=end_time)
        all_signals.extend(sigs)

    all_signals.sort(key=lambda x: x['time'], reverse=True)
    limited = all_signals[:limit]
    return jsonify({"signals": limited})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
