# app.py
import os
import time
import requests
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory, abort

# ---------------------------
# CONFIG
# ---------------------------
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_KEY", "EJFK7GKQO9DWXDS8")

EMA_SHORT = 20
EMA_LONG  = 50
RSI_LEN   = 14
ATR_LEN   = 14

DAILY_SIGNAL_CAP = 10

# cache
CACHE = {}
CACHE_TTL = 30  # seconds

# base dir + Flask app with absolute template/static paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR
)

# ---------------------------
# Helpers - pure Python implementations
# ---------------------------

def parse_time_string(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None

def fetch_fx_intraday_list(base, quote, interval="1min", outputsize="compact"):
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
    except Exception:
        return None

    key = f"Time Series FX ({interval})"
    if key not in data:
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

    records.sort(key=lambda x: x['time'])
    CACHE[key_cache] = {'ts': now_ts, 'data': records}
    return records


def compute_ema_list(prices, span):
    n = len(prices)
    ema = [None] * n
    if n < span:
        return ema
    sma = sum(prices[:span]) / span
    ema[span-1] = sma
    k = 2.0 / (span + 1)
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

    avg_gain = sum(gains[1:period+1]) / period
    avg_loss = sum(losses[1:period+1]) / period
    rs = float('inf') if avg_loss == 0 else (avg_gain / avg_loss)
    rsi[period] = 100 - (100 / (1 + rs))

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / (avg_loss if avg_loss != 0 else 1e-10)
        rsi[i] = 100 - (100 / (1 + rs))
    return rsi


def compute_atr_list(records, period=14):
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
    initial_trs = [tr[i] for i in range(1, period+1) if tr[i] is not None]
    if len(initial_trs) < period:
        return atr
    atr_val = sum(initial_trs) / period
    atr[period] = atr_val
    for i in range(period + 1, n):
        atr_val = ((atr_val * (period - 1)) + tr[i]) / period
        atr[i] = atr_val
    return atr


def compute_signals(records, base, quote):
    if not records or len(records) < max(EMA_LONG, RSI_LEN, ATR_LEN) + 5:
        return []

    closes = [r['close'] for r in records]
    ema_s = compute_ema_list(closes, EMA_SHORT)
    ema_l = compute_ema_list(closes, EMA_LONG)

    signals = []
    n = len(records)
    for i in range(n-1, 0, -1):
        if ema_s[i] is None or ema_l[i] is None or ema_s[i-1] is None or ema_l[i-1] is None:
            continue

        golden_cross = (ema_s[i] > ema_l[i]) and (ema_s[i-1] <= ema_l[i-1])
        death_cross  = (ema_s[i] < ema_l[i]) and (ema_s[i-1] >= ema_l[i-1])

        px = records[i]['close']
        if golden_cross:
            signals.append({
                "time": records[i]['time'].strftime("%H:%M"),
                "pair": f"{base}/{quote}",
                "type": "BUY",
                "price": round(px, 6),
                "reason": f"EMA{EMA_SHORT}>{EMA_LONG} cross"
            })

        if death_cross:
            signals.append({
                "time": records[i]['time'].strftime("%H:%M"),
                "pair": f"{base}/{quote}",
                "type": "SELL",
                "price": round(px, 6),
                "reason": f"EMA{EMA_SHORT}<{EMA_LONG} cross"
            })

        if len(signals) >= DAILY_SIGNAL_CAP:
            break

    signals.reverse()
    return signals



def within_time_window(ts_hhmm, start_hhmm, end_hhmm):
    from datetime import time as dtime
    def to_time(s):
        return dtime(hour=int(s[:2]), minute=int(s[3:5]))
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
    # render_template will use the absolute template_folder set above
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

            records = fetch_fx_intraday_list(base, quote, interval="1min", outputsize="compact")
            if not records:
                continue

            sigs = compute_signals(records, base, quote)
            all_signals.extend(sigs)

            if idx < len(pairs) - 1:
                time.sleep(1)

        result = filter_signals(all_signals, start_time, end_time, direction)
        result = result[:DAILY_SIGNAL_CAP]
        return jsonify(result), 200
    except Exception as e:
        # log full traceback
        print("Exception in /signals:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Debug & utility routes (protected)
# ---------------------------

# store last exception text (updated by error handler)
app.config['LAST_EXCEPTION_TEXT'] = None

@app.errorhandler(Exception)
def handle_all_exceptions(e):
    app.config['LAST_EXCEPTION_TEXT'] = traceback.format_exc()
    print("=== FULL TRACEBACK (captured) ===")
    print(app.config['LAST_EXCEPTION_TEXT'])
    print("=== END TRACEBACK ===")
    return "Internal server error (debug captured)", 500

# protected token for debug access (set via ENV in Render)
DEBUG_TOKEN = os.getenv("DEV_DEBUG_TOKEN", "dev_debug_token_please_change")

@app.route("/_last_error")
def last_error():
    token = request.args.get("token", "")
    if token != DEBUG_TOKEN:
        abort(403)
    txt = app.config.get('LAST_EXCEPTION_TEXT')
    if not txt:
        return "No exception captured yet."
    return "<pre style='white-space:pre-wrap; font-size:12px;'>%s</pre>" % (txt,)

@app.route("/_ls")
def _list_templates():
    base = app.root_path
    tpl_dir = os.path.join(base, "templates")
    out = {"app_root": base, "templates_path": tpl_dir, "exists": os.path.isdir(tpl_dir), "files": []}
    if os.path.isdir(tpl_dir):
        try:
            names = sorted(os.listdir(tpl_dir))
            out["files"] = names
        except Exception as e:
            out["error_listdir"] = str(e)
    print("===== TEMPLATES DEBUG =====")
    print(out)
    print("===== END TEMPLATES DEBUG =====")
    html = "<h3>Templates debug</h3>"
    html += f"<p>App root: <code>{out['app_root']}</code></p>"
    html += f"<p>Templates dir exists: {out['exists']}</code></p>"
    html += "<ul>"
    for f in out["files"]:
        html += f"<li>{f}</li>"
    html += "</ul>"
    if "error_listdir" in out:
        html += f"<pre>{out['error_listdir']}</pre>"
    return html

@app.route("/test_index")
def test_index():
    # direct send file (bypasses Jinja) to confirm file presence
    path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.isfile(path):
        return send_from_directory(TEMPLATES_DIR, "index.html")
    return f"index.html NOT FOUND at {path}", 404


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    # Development server
    app.run(host="0.0.0.0", port=5000, debug=True)
