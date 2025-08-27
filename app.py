from flask import Flask, jsonify, request, send_file
import requests, time

app = Flask(__name__)
API_KEY = "34d108698a5044abb24fbdee2ef0cd91"
# Simple cache: { pair_symbol: {'timestamp': <epoch>, 'values': [...] } }
cache = {}

def fetch_data(pair):
    """
    Fetch intraday (1min) OHLC data for a given forex pair from Twelve Data API.
    Caches results for 30 seconds to avoid rate limits.
    """
    now = time.time()
    # Return cached data if fresh
    if pair in cache and now - cache[pair]['timestamp'] < 30:
        return cache[pair]['values']
    # Otherwise, fetch from Twelve Data
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={pair}&interval=1min&outputsize=100&apikey={API_KEY}"
    )
    try:
        res = requests.get(url)
        data = res.json()
    except Exception as e:
        print(f"Error fetching {pair}: {e}")
        return []
    values = data.get("values", [])
    values.reverse()  # ensure chronological order (oldest first)
    cache[pair] = {'timestamp': now, 'values': values}
    return values

@app.route('/')
def index():
    # Serve the index.html (assumed to be in the same directory)
    return send_file('index.html')

@app.route('/signals', methods=['POST'])
def signals():
    """
    POST JSON: {"pairs": ["EUR/USD", "GBP/USD", ...]}
    Returns JSON with up to 10 most recent BUY/SELL signals for the requested pairs.
    """
    req = request.get_json()
    if not req or 'pairs' not in req:
        return jsonify({"error": "No 'pairs' field in request"}), 400
    pairs = req.get('pairs', [])
    all_signals = []

    for pair in pairs:
        values = fetch_data(pair)
        if not values:
            continue  # skip if no data
        n = len(values)
        if n < 70:
            continue  # not enough data for EMA70/RSI14/ATR14
        
        # Extract OHLC arrays
        closes = [float(v['close']) for v in values]
        highs  = [float(v['high']) for v in values]
        lows   = [float(v['low']) for v in values]
        times  = [v['datetime'] for v in values]

        # Calculate EMA(20) and EMA(70)
        ema20 = [None]*n
        ema70 = [None]*n
        # Initial SMA for first EMA values
        ema20[19] = sum(closes[:20]) / 20.0
        for i in range(20, n):
            alpha = 2 / (20 + 1)
            ema20[i] = closes[i]*alpha + ema20[i-1]*(1-alpha)
        ema70[69] = sum(closes[:70]) / 70.0
        for i in range(70, n):
            alpha = 2 / (70 + 1)
            ema70[i] = closes[i]*alpha + ema70[i-1]*(1-alpha)

        # Calculate RSI(14) using Wilder's smoothing
        rsi = [None]*n
        # First average gain/loss
        gains = 0.0
        losses = 0.0
        for i in range(1, 15):
            delta = closes[i] - closes[i-1]
            if delta > 0:
                gains += delta
            else:
                losses -= delta  # negative delta
        avg_gain = gains / 14.0
        avg_loss = losses / 14.0
        # First RSI value at index 14
        if avg_loss == 0:
            rsi[14] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[14] = 100.0 - (100.0 / (1 + rs))
        # Remaining RSI values
        for i in range(15, n):
            delta = closes[i] - closes[i-1]
            gain = delta if delta > 0 else 0
            loss = -delta if delta < 0 else 0
            avg_gain = (avg_gain * 13 + gain) / 14.0
            avg_loss = (avg_loss * 13 + loss) / 14.0
            if avg_loss == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - (100.0 / (1 + rs))

        # Calculate ATR(14)
        atr = [None]*n
        tr14_sum = 0.0
        # First 14 True Range values
        for i in range(1, 15):
            prev = closes[i-1]
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - prev),
                abs(prev - lows[i])
            )
            tr14_sum += tr
        atr[14] = tr14_sum / 14.0
        for i in range(15, n):
            prev = closes[i-1]
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - prev),
                abs(prev - lows[i])
            )
            atr[i] = (atr[i-1] * 13 + tr) / 14.0

        # Identify crossover signals
        for i in range(70, n):
            if ema20[i-1] is None or ema70[i-1] is None or rsi[i] is None:
                continue
            # Bullish crossover: EMA20 crosses above EMA70
            if ema20[i-1] < ema70[i-1] and ema20[i] > ema70[i]:
                # Confirm with RSI oversold (<30) indicating reversal potential
                if rsi[i] < 30:
                    entry = closes[i]
                    atr_val = atr[i] if atr[i] is not None else 0.0
                    sl = entry - atr_val
                    tp = entry + 2 * atr_val
                    all_signals.append({
                        "time": times[i],
                        "pair": pair,
                        "signal": "BUY",
                        "entry_price": entry,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "reason": "20 EMA crossed above 70 EMA + RSI oversold confirmation"
                    })
            # Bearish crossover: EMA20 crosses below EMA70
            elif ema20[i-1] > ema70[i-1] and ema20[i] < ema70[i]:
                if rsi[i] > 70:
                    entry = closes[i]
                    atr_val = atr[i] if atr[i] is not None else 0.0
                    sl = entry + atr_val
                    tp = entry - 2 * atr_val
                    all_signals.append({
                        "time": times[i],
                        "pair": pair,
                        "signal": "SELL",
                        "entry_price": entry,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "reason": "20 EMA crossed below 70 EMA + RSI overbought confirmation"
                    })

    # Sort signals by time (newest first) and limit to 10
    all_signals.sort(key=lambda x: x["time"], reverse=True)
    limited = all_signals[:10]
    return jsonify({"signals": limited})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

