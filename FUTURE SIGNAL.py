import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, time as dtime
from flask import Flask, request, jsonify, render_template, send_from_directory

# ---------------------------
# CONFIG
# ---------------------------
# আপনার Alpha Vantage API key .env/Environment Variable হিসেবে সেট করে দিন:
# Linux/macOS উদাহরণ: export ALPHA_VANTAGE_KEY=EJFK7GKQO9DWXDS8
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_KEY", "EJFK7GKQO9DWXDS8")

# ইন্ডিকেটর সেটিং (আপনি চাইলে টিউন করতে পারেন)
EMA_SHORT = 20
EMA_LONG  = 50
RSI_LEN   = 14
ATR_LEN   = 14

# প্রতিদিন সর্বোচ্চ কতটি সিগন্যাল দেখাতে চান
DAILY_SIGNAL_CAP = 10

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------------------
# Helper Functions
# ---------------------------

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-10))
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    # True Range
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

def fetch_fx_intraday(base, quote, interval="1min", outputsize="compact"):
    """
    Alpha Vantage FX_INTRADAY থেকে 1-মিনিট ক্যান্ডল ডেটা আনে।
    FX ডেটায় সাধারণত ভলিউম থাকে না; তাই আমরা OHLC নিয়েই কাজ করবো।
    """
    url = (
        "https://www.alphavantage.co/query?"
        "function=FX_INTRADAY"
        f"&from_symbol={base}&to_symbol={quote}"
        f"&interval={interval}"
        f"&apikey={ALPHA_KEY}"
        f"&outputsize={outputsize}"
    )
    resp = requests.get(url, timeout=30)
    data = resp.json()
    key = f"Time Series FX ({interval})"
    if key not in data:
        # API limit বা invalid হলে graceful fallback
        return None

    ts = data[key]
    df = pd.DataFrame.from_dict(ts, orient="index")
    # Alpha Vantage FX keys: '1. open', '2. high', '3. low', '4. close'
    df = df.rename(columns={
        '1. open': 'open',
        '2. high': 'high',
        '3. low' : 'low',
        '4. close': 'close'
    })
    df = df.astype(float)
    # Index to datetime (UTC-ish string)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def compute_signals(df, base, quote):
    """
    High-quality but conservative signals: EMA cross + RSI confirmation (+ ATR for SL/TP).
    লক্ষ্য: ফাস্ট EMA/স্লো EMA ক্রসিং, RSI extremes দিয়ে ফিল্টার।
    """
    if df is None or len(df) < max(EMA_LONG, RSI_LEN, ATR_LEN) + 5:
        return []

    # Indicators
    df['ema_s'] = df['close'].ewm(span=EMA_SHORT, adjust=False).mean()
    df['ema_l'] = df['close'].ewm(span=EMA_LONG, adjust=False).mean()
    df['rsi']   = rsi(df['close'], RSI_LEN)
    df['atr']   = atr(df[['high','low','close']].copy(), ATR_LEN)

    # Shifted for cross detection
    df['ema_s_prev'] = df['ema_s'].shift(1)
    df['ema_l_prev'] = df['ema_l'].shift(1)

    signals = []
    # iterate from most recent to older to collect robust, fresh signals
    for ts in df.index[::-1]:
        row = df.loc[ts]
        prev = df.shift(1).loc[ts]
        # EMA golden/death cross detection
        golden_cross = (row['ema_s'] > row['ema_l']) and (prev['ema_s'] <= prev['ema_l'])
        death_cross  = (row['ema_s'] < row['ema_l']) and (prev['ema_s'] >= prev['ema_l'])

        # Confirmation filters (tight to reduce noise on 1-min)
        # Buy: RSI < 35 এবং close > ema_l (trend bias)
        # Sell: RSI > 65 এবং close < ema_l
        # Extremely cautious: skip if ATR not ready
        if pd.isna(row['atr']) or row['atr'] == 0:
            continue

        px = float(row['close'])
        pip_buffer = float(row['atr'])  # ATR হিসেবে SL keep
        # TP ≈ 1.25R
        tp_buffer = 1.25 * pip_buffer

        if golden_cross and (row['rsi'] < 35) and (row['close'] > row['ema_l']):
            signals.append({
                "time": ts.strftime("%H:%M"),
                "pair": f"{base}/{quote}",
                "type": "BUY",
                "price": round(px, 6),
                "sl": round(px - pip_buffer, 6),
                "tp": round(px + tp_buffer, 6),
                "reason": f"EMA{EMA_SHORT}>{EMA_LONG} cross + RSI({RSI_LEN})={round(row['rsi'],1)} < 35; trend above EMA_long"
            })

        if death_cross and (row['rsi'] > 65) and (row['close'] < row['ema_l']):
            signals.append({
                "time": ts.strftime("%H:%M"),
                "pair": f"{base}/{quote}",
                "type": "SELL",
                "price": round(px, 6),
                "sl": round(px + pip_buffer, 6),
                "tp": round(px - tp_buffer, 6),
                "reason": f"EMA{EMA_SHORT}<{EMA_LONG} cross + RSI({RSI_LEN})={round(row['rsi'],1)} > 65; trend below EMA_long"
            })

        # stop if we already have enough per pair
        if len(signals) >= DAILY_SIGNAL_CAP:
            break

    # reverse to chronological (oldest->newest)
    signals.reverse()
    return signals

def within_time_window(ts_hhmm, start_hhmm, end_hhmm):
    """ts_hhmm, start_hhmm, end_hhmm are 'HH:MM' strings."""
    to_time = lambda s: dtime(hour=int(s[:2]), minute=int(s[3:5]))
    t  = to_time(ts_hhmm)
    s  = to_time(start_hhmm) if start_hhmm else dtime(0,0)
    e  = to_time(end_hhmm)   if end_hhmm   else dtime(23,59)
    # সাধারণ window (same-day) ধরে নিয়েছি
    return (t >= s) and (t <= e)

def filter_signals(signals, start_time, end_time, direction):
    # time window
    filtered = [s for s in signals if within_time_window(s["time"], start_time, end_time)]
    # direction filter
    if direction == "CALL":
        filtered = [s for s in filtered if s["type"] == "BUY"]
    elif direction == "PUT":
        filtered = [s for s in filtered if s["type"] == "SELL"]
    # লিমিট
    return filtered[:DAILY_SIGNAL_CAP]

# ---------------------------
# Routes
# ---------------------------

@app.route("/")
def index():
    # templates/index.html রেন্ডার করবে
    return render_template("index.html")

@app.route("/signals", methods=["POST"])
def signals():
    """
    POST JSON: { pairs: ["EUR/USD","USD/JPY",...],
                 startTime: "HH:MM",
                 endTime: "HH:MM",
                 signalDirection: "CALL"|"PUT"|"BOTH" }
    """
    try:
        data = request.get_json(force=True)
        pairs = data.get("pairs", [])
        start_time = data.get("startTime", "")
        end_time = data.get("endTime", "")
        direction = data.get("signalDirection", "BOTH")

        all_signals = []
        # Alpha Vantage ফ্রি প্ল্যানে rate-limit আছে (প্রতি মিনিটে ৫টি কল)
        # তাই pair গুলোতে একটু থ্রটলিং করতে চাইলে এখানে sleep রাখতে পারেন।
        for i, pair in enumerate(pairs):
            base, quote = pair.split("/")
            df = fetch_fx_intraday(base, quote, interval="1min", outputsize="compact")
            sigs = compute_signals(df, base, quote)
            all_signals.extend(sigs)

            # রেট-লিমিট ফ্রেন্ডলি (optional):
            if i < len(pairs) - 1:
                time.sleep(1)  # একটু দেরি

        # ফাইনাল ফিল্টারিং
        result = filter_signals(all_signals, start_time, end_time, direction)
        # সর্বোচ্চ 10টি
        result = result[:DAILY_SIGNAL_CAP]
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    # ডেভেলপমেন্টে রান:
    # export FLASK_ENV=development && python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)
