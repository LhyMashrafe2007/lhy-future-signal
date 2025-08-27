# FUTURE SIGNAL.py
import os
import time
import requests
from datetime import datetime
from flask import Flask, request, jsonify, render_template
# --- Debug helper: capture last exception and make accessible temporarily ---
import traceback
from flask import abort
# ---------------------------
# CONFIG
# ---------------------------
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_KEY", "EJFK7GKQO9DWXDS8")

# Indicator params (adjustable)
EMA_SHORT = 20
EMA_LONG  = 50
RSI_LEN   = 14
ATR_LEN   = 14

DAILY_SIGNAL_CAP = 10

# Simple in-memory cache to reduce API calls (key: "BASE/QUOTE")
CACHE = {}
CACHE_TTL = 30  # seconds

app = Flask(__name__, template_folder="templates", static_folder="static")


# ---------------------------
# Helpers - pure Python implementations
# ---------------------------

def parse_time_string(s):
    # AlphaVantage timestamps usually "YYYY-MM-DD HH:MM:SS"
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        try:
            # sometimes 'T' separator or timezone, try fromisoformat
            return datetime.fromisoformat(s)
        except Exception:
            return None

def fetch_fx_intraday_list(base, quote, interval="1min", outputsize="compact"):
    """
    Returns a list of candles sorted oldest->newest:
    [ { 'time': datetime, 'open':float, 'high':float, 'low':float, 'close':float }, ... ]
    """
    key_cache = f"{base}/{quote}/{interval}"
    now_ts = time.time()
    if key_cache in CACHE and now_ts - CACHE[key_cache]['ts'] < CACHE_TTL:
        return CACHE[key_cache]['data']

    url = (
        "https://www.alphavantage.co/query?"
        "function=FX_INTRADAY"
        f"&from_symbol={base}&to_symbol={quote}"
        f"&interval={interval}&outputsize={outputsize}&apikey={ALPHA_KEY}"
    )
    try:
        resp = requests.get(url, timeout=25)
        data = resp.json()
    except Exception as e:
        return None

    key = f"Time Series FX ({interval})"
    if key not in data:
        # sometimes response might contain error message or rate limit notice
        return None

    ts = data[key]
    records = []
    for tstr, vals in ts.items():
        dt = parse_time_string(tstr)
        if not dt:
            continue
        try:
            o = float(vals.get("1. open"))
            h = float(vals.get("2. high"))
            l = float(vals.get("3. low"))
            c = float(vals.get("4. close"))
        except Exception:
            continue
        records.append({'time': dt, 'open': o, 'high': h, 'low': l, 'close': c})

    # sort ascending (oldest -> newest)
    records.sort(key=lambda x: x['time'])
    # cache
    CACHE[key_cache] = {'ts': now_ts, 'data': records}
    return records


def compute_ema_list(prices, span):
    n = len(prices)
    ema = [None] * n
    if n < span:
        return ema
    # initial SMA
    sma = sum(prices[:span]) / span
    ema[span-1] = sma
    k = 2.0 / (span + 1)
    # compute subsequent
    for i in range(span, n):
        ema[i] = (prices[i] - ema[i-1]) * k + ema[i-1]
    return ema


def compute_rsi_list(prices, period=14):
    n = len(prices)
    rsi = [None] * n
    if n < period + 1:
        return rsi
    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        diff = prices[i] - prices[i-1]
        gains[i] = diff if diff > 0 else 0.0
        losses[i] = -diff if diff < 0 else 0.0

    # initial avg gain/loss
    avg_gain = sum(gains[1:period+1]) / period
    avg_loss = sum(losses[1:period+1]) / period
    if avg_loss == 0:
        rs = float('inf')
    else:
        rs = avg_gain / avg_loss
    rsi[period] = 100 - (100 / (1 + rs))

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / (avg_loss if avg_loss != 0 else 1e-10)
        rsi[i] = 100 - (100 / (1 + rs))
    return rsi


def compute_atr_list(records, period=14):
    # records is list of dicts with keys high, low, close
    n = len(records)
    if n < period + 1:
        return [None] * n
    tr = [None] * n
    for i in range(1, n):
        high = records[i]['high']
        low = records[i]['low']
        prev_close = records[i-1]['close']
        tr[i] = max(high - low, abs(high - prev_close), abs(low - prev_close))

    atr = [None] * n
    # initial ATR is average of first `period` TR values (from 1..period)
    initial_trs = [tr[i] for i in range(1, period+1) if tr[i] is not None]
    if len(initial_trs) < period:
        return atr
    atr_val = sum(initial_trs) / period
    atr[period] = atr_val
    for i in range(period + 1, n):
        atr_val = ( (atr_val * (period - 1)) + tr[i] ) / period
        atr[i] = atr_val
    return atr


def compute_signals(records, base, quote):
    """
    records: list of candle dicts sorted oldest->newest
    returns list of signal dicts (time, pair, type, price, sl, tp, reason)
    """
    if not records or len(records) < max(EMA_LONG, RSI_LEN, ATR_LEN) + 5:
        return []

    closes = [r['close'] for r in records]
    highs = [r['high'] for r in records]
    lows = [r['low'] for r in records]

    ema_s = compute_ema_list(closes, EMA_SHORT)
    ema_l = compute_ema_list(closes, EMA_LONG)
    rsi = compute_rsi_list(closes, RSI_LEN)
    # build minimal structure for atr computation
    recs_min = [{'high': highs[i], 'low': lows[i], 'close': closes[i]} for i in range(len(records))]
    atr = compute_atr_list(recs_min, ATR_LEN)

    signals = []
    n = len(records)
    # iterate from newest backward to find freshest signals
    for i in range(n-1, 0, -1):
        # need previous index for cross detection
        if i-1 < 0:
            continue
        if ema_s[i] is None or ema_l[i] is None or ema_s[i-1] is None or ema_l[i-1] is None:
            continue
        # cross detection
        golden_cross = (ema_s[i] > ema_l[i]) and (ema_s[i-1] <= ema_l[i-1])
        death_cross  = (ema_s[i] < ema_l[i]) and (ema_s[i-1] >= ema_l[i-1])

        # require rsi and atr present
        if rsi[i] is None or atr[i] is None:
            continue

        px = records[i]['close']
        pip_buffer = atr[i]  # use ATR as buffer (approx)
        tp_buffer = 1.25 * pip_buffer

        if golden_cross and (rsi[i] < 35) and (records[i]['close'] > ema_l[i]):
            signals.append({
                "time": records[i]['time'].strftime("%H:%M"),
                "pair": f"{base}/{quote}",
                "type": "BUY",
                "price": round(px, 6),
                "sl": round(px - pip_buffer, 6),
                "tp": round(px + tp_buffer, 6),
                "reason": f"EMA{EMA_SHORT}>{EMA_LONG} cross + RSI({RSI_LEN})={round(rsi[i],1)} < 35"
            })

        if death_cross and (rsi[i] > 65) and (records[i]['close'] < ema_l[i]):
            signals.append({
                "time": records[i]['time'].strftime("%H:%M"),
                "pair": f"{base}/{quote}",
                "type": "SELL",
                "price": round(px, 6),
                "sl": round(px + pip_buffer, 6),
                "tp": round(px - tp_buffer, 6),
                "reason": f"EMA{EMA_SHORT}<{EMA_LONG} cross + RSI({RSI_LEN})={round(rsi[i],1)} > 65"
            })

        # stop if enough per pair
        if len(signals) >= DAILY_SIGNAL_CAP:
            break

    # reverse chronological order (oldest->newest)
    signals.reverse()
    return signals


def within_time_window(ts_hhmm, start_hhmm, end_hhmm):
    from datetime import time as dtime
    def to_time(s):
        return dtime(hour=int(s[:2]), minute=int(s[3:5]))
    t = None
    try:
        t = to_time(ts_hhmm)
    except Exception:
        return True
    s = to_time(start_hhmm) if start_hhmm else dtime(0,0)
    e = to_time(end_hhmm) if end_hhmm else dtime(23,59)
    return (t >= s) and (t <= e)


def filter_signals(signals, start_time, end_time, direction):
    filtered = [s for s in signals if within_time_window(s["time"], start_time, end_time)]
    if direction == "CALL":
        filtered = [s for s in filtered if s["type"] == "BUY"]
    elif direction == "PUT":
        filtered = [s for s in filtered if s["type"] == "SELL"]
    return filtered[:DAILY_SIGNAL_CAP]


# ---------------------------
# Routes
# ---------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/signals", methods=["POST"])
def signals_route():
    try:
        data = request.get_json(force=True)
        pairs = data.get("pairs", [])
        start_time = data.get("startTime", "")
        end_time = data.get("endTime", "")
        direction = data.get("signalDirection", "BOTH")

        all_signals = []

        for idx, pair in enumerate(pairs):
            try:
                base, quote = pair.split("/")
            except Exception:
                continue
            # fetch (with caching)
            records = fetch_fx_intraday_list(base, quote, interval="1min", outputsize="compact")
            if records is None or len(records) == 0:
                # skip if no data
                continue

            sigs = compute_signals(records, base, quote)
            all_signals.extend(sigs)

            # be gentle with API (throttle)
            if idx < len(pairs) - 1:
                time.sleep(1)

        # final filtering by time/direction & limit
        result = filter_signals(all_signals, start_time, end_time, direction)
        result = result[:DAILY_SIGNAL_CAP]
        return jsonify(result), 200
    except Exception as e:
        # return error for frontend to show
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    # Dev server
    app.run(host="0.0.0.0", port=5000, debug=True)
# Store last exception text
app.config['LAST_EXCEPTION_TEXT'] = None

@app.errorhandler(Exception)
def handle_all_exceptions(e):
    # Save full traceback to app config (for temporary debugging only)
    app.config['LAST_EXCEPTION_TEXT'] = traceback.format_exc()
    # Also print to stdout so Render logs will have it
    print("=== FULL TRACEBACK (captured) ===")
    print(app.config['LAST_EXCEPTION_TEXT'])
    print("=== END TRACEBACK ===")
    # Return minimal 500 (frontend still sees 500)
    return "Internal server error (debug captured)", 500

# Debug route to read last exception (protected by a token)
DEBUG_TOKEN = os.getenv("DEV_DEBUG_TOKEN", "dev_debug_token_please_change")

@app.route("/_last_error")
def last_error():
    token = request.args.get("token", "")
    if token != DEBUG_TOKEN:
        abort(403)
    txt = app.config.get('LAST_EXCEPTION_TEXT')
    if not txt:
        return "No exception captured yet."
    # Return as plain text
    return "<pre style='white-space:pre-wrap; font-size:12px;'>%s</pre>" % (txt,)