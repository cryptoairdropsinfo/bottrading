import time
import math
import requests
import pandas as pd

# =========================
# CONFIG
# =========================
TIMEFRAMES = ["3m", "1h"]          # Ambos TFs — señal solo si coinciden
LIMIT = 300
CHECK_EVERY_SECONDS = 60

# ── Indicadores ──────────────────────────────────────
RSI_PERIOD      = 14
ATR_PERIOD      = 14
EMA_FAST        = 9                 # EMA rápida (cruce)
EMA_MED         = 21                # EMA media  (cruce)
EMA_TREND       = 50                # EMA tendencia principal
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL_P   = 9
VOL_MA_PERIOD   = 20
COOLDOWN_CANDLES = 6

MIN_VOLUME_RATIO    = 1.3           # volumen mínimo vs media 20
ATR_STOP_MULTIPLIER = 1.5           # SL = ATR × 1.5
TP_RR_RATIO         = 2.0           # TP = riesgo × 2 (ratio 2:1)

MAX_OPEN_SIGNALS = 10

# ── Backend (mismo que tu bot actual) ────────────────
LOVABLE_BASE_URL = "https://fsjkfaallagpkuoliizg.supabase.co/functions/v1"
BOT_API_KEY      = "ragery_bot_secure_982347234"

# ── Binance USDⓈ-M Futures ───────────────────────────
BINANCE_BASE_URL  = "https://fapi.binance.com"
KLINES_URL        = f"{BINANCE_BASE_URL}/fapi/v1/klines"
PRICE_URL         = f"{BINANCE_BASE_URL}/fapi/v1/ticker/price"
EXCHANGE_INFO_URL = f"{BINANCE_BASE_URL}/fapi/v1/exchangeInfo"
TICKER_24H_URL    = f"{BINANCE_BASE_URL}/fapi/v1/ticker/24hr"

# ── 10 monedas iniciales ─────────────────────────────
TOP_10_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "BNBUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "DOTUSDT",
]

# ── Estado en runtime ────────────────────────────────
last_alerted     = {}   # symbol -> candle_id último alerta
last_signal_idx  = {}   # symbol -> índice de vela (cooldown)
open_signals     = {}   # signal_id -> datos


# =========================
# HELPERS
# =========================
def symbol_to_display(symbol: str) -> str:
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}/USDT:USDT"
    return symbol


def safe_get(url: str, params=None, retries=1, label="req") -> any:
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[{label}] intento {attempt+1}: {e}")
            if attempt < retries:
                time.sleep(2)
    return None


# =========================
# MARKET DATA (Binance)
# =========================
def fetch_klines(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame | None:
    data = safe_get(KLINES_URL, params={"symbol": symbol, "interval": interval, "limit": limit},
                    retries=1, label=f"{symbol}/{interval} klines")
    if not isinstance(data, list) or len(data) < 60:
        return None

    rows = []
    for k in data:
        try:
            rows.append([int(k[0]), float(k[1]), float(k[2]),
                         float(k[3]), float(k[4]), float(k[5])])
        except Exception:
            continue

    if len(rows) < 60:
        return None

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def get_current_price(symbol: str) -> float | None:
    data = safe_get(PRICE_URL, params={"symbol": symbol}, retries=1, label=f"{symbol} price")
    try:
        return float(data["price"])
    except Exception:
        return None


# =========================
# INDICADORES
# =========================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, math.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift(1)).abs()
    lc  = (df["low"]  - df["close"].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # EMAs
    df["ema9"]   = df["close"].ewm(span=EMA_FAST,  adjust=False).mean()
    df["ema21"]  = df["close"].ewm(span=EMA_MED,   adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=EMA_TREND, adjust=False).mean()

    # MACD
    df["macd"], df["macd_sig"], df["macd_hist"] = compute_macd(
        df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL_P
    )

    # RSI y ATR
    df["rsi"]    = compute_rsi(df["close"], RSI_PERIOD)
    df["atr"]    = compute_atr(df, ATR_PERIOD)

    # Volumen
    df["vol_ma"] = df["volume"].rolling(VOL_MA_PERIOD).mean()

    return df


# =========================
# ESTRATEGIA EMA/MACD
# =========================
def get_signal_on_tf(df: pd.DataFrame) -> str | None:
    """
    Evalúa la última vela cerrada.
    LONG  : EMA9>EMA21>EMA50  AND  MACD cruza al alza  AND  RSI 45-70  AND  vol > media
    SHORT : EMA9<EMA21<EMA50  AND  MACD cruza a la baja AND  RSI 30-55  AND  vol > media
    Devuelve 'LONG', 'SHORT' o None.
    """
    row  = df.iloc[-2]   # última vela cerrada
    prev = df.iloc[-3]   # penúltima (para detectar cruce)

    vol_ok = (
        pd.notna(row["vol_ma"]) and
        row["vol_ma"] > 0 and
        row["volume"] >= row["vol_ma"] * MIN_VOLUME_RATIO
    )

    # Verificar que los indicadores están disponibles
    required = ["ema9", "ema21", "ema50", "macd", "macd_sig", "rsi", "atr"]
    if any(pd.isna(row[c]) for c in required):
        return None

    # ── LONG ─────────────────────────────────────────
    ema_bull    = row["ema9"] > row["ema21"] > row["ema50"]
    macd_cross_up = (prev["macd"] <= prev["macd_sig"]) and (row["macd"] > row["macd_sig"])
    rsi_ok_long = 45 <= row["rsi"] <= 70
    candle_bull = row["close"] > row["open"]

    if ema_bull and macd_cross_up and rsi_ok_long and vol_ok and candle_bull:
        return "LONG"

    # ── SHORT ─────────────────────────────────────────
    ema_bear     = row["ema9"] < row["ema21"] < row["ema50"]
    macd_cross_dn= (prev["macd"] >= prev["macd_sig"]) and (row["macd"] < row["macd_sig"])
    rsi_ok_short = 30 <= row["rsi"] <= 55
    candle_bear  = row["close"] < row["open"]

    if ema_bear and macd_cross_dn and rsi_ok_short and vol_ok and candle_bear:
        return "SHORT"

    return None


def get_confirmed_signal(symbol: str) -> tuple[str | None, dict | None, str]:
    """
    Confirmación multi-TF: 3m + 1H deben coincidir en dirección.
    Devuelve (side, row_3m, strategy_name) o (None, None, '').
    """
    # TF primario: 3m (entrada)
    df_3m = fetch_klines(symbol, "3m",  limit=LIMIT)
    # TF filtro:   1H  (tendencia)
    df_1h = fetch_klines(symbol, "1h",  limit=200)

    if df_3m is None or df_1h is None:
        return None, None, ""

    df_3m = enrich(df_3m)
    df_1h = enrich(df_1h)

    sig_3m = get_signal_on_tf(df_3m)
    sig_1h = get_signal_on_tf(df_1h)

    # Ambos TF deben dar la misma dirección
    if sig_3m is None or sig_1h is None:
        return None, None, ""
    if sig_3m != sig_1h:
        return None, None, ""

    row_3m = df_3m.iloc[-2]
    name   = f"EMA/MACD Trend {'Long' if sig_3m == 'LONG' else 'Short'} (3m+1H confirm)"
    return sig_3m, row_3m, name


# =========================
# RISK MANAGEMENT (2:1)
# =========================
def calculate_levels(side: str, row: pd.Series) -> dict | None:
    entry = float(row["close"])
    atr   = float(row["atr"])

    if atr <= 0 or math.isnan(atr):
        return None

    sl_dist = atr * ATR_STOP_MULTIPLIER
    tp_dist = sl_dist * TP_RR_RATIO

    if side == "LONG":
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist

    return {
        "entry_price":       entry,
        "stop_loss":         sl,
        "take_profit":       tp,
        "risk_reward_ratio": TP_RR_RATIO,
        "atr_value":         atr,
    }


# =========================
# BACKEND — CREAR SEÑAL
# =========================
def send_signal(symbol: str, side: str, strategy_name: str, row: pd.Series):
    if len(open_signals) >= MAX_OPEN_SIGNALS:
        print(f"[SKIP] Máximo de señales abiertas ({MAX_OPEN_SIGNALS}) alcanzado.")
        return

    levels = calculate_levels(side, row)
    if not levels:
        return

    vol_ma    = float(row["vol_ma"]) if pd.notna(row["vol_ma"]) and row["vol_ma"] > 0 else 0
    vol_ratio = float(row["volume"]) / vol_ma if vol_ma > 0 else 0

    payload = {
        "symbol":             symbol_to_display(symbol),
        "side":               side.lower(),
        "timeframe":          "3m+1H",
        "strategy_name":      strategy_name,
        "entry_price":        levels["entry_price"],
        "stop_loss":          levels["stop_loss"],
        "take_profit":        levels["take_profit"],
        "risk_reward_ratio":  levels["risk_reward_ratio"],
        "rsi_value":          float(row["rsi"]),
        "volume_ratio":       round(vol_ratio, 2),
        "atr_value":          levels["atr_value"],
        "signal_time":        row["timestamp"].isoformat(),
        "status":             "active",
        "signal_source":      "bot",
        "notes":              f"EMA9/21/50 + MACD cruce | ATR SL {ATR_STOP_MULTIPLIER}x | TP 2:1 | Confirm 3m+1H",
    }

    headers = {
        "Content-Type": "application/json",
        "x-bot-api-key": BOT_API_KEY,
    }

    try:
        r = requests.post(
            f"{LOVABLE_BASE_URL}/bot-signals",
            json=payload, headers=headers, timeout=20
        )
        print(f"[SIGNAL] {symbol} {side} → {r.status_code} {r.text}")

        if r.status_code not in (200, 201):
            return

        data      = r.json()
        signal_id = None

        if isinstance(data, dict):
            signal_id = (
                data.get("id") or
                data.get("signal_id") or
                (data.get("signal") or {}).get("id") or
                (data.get("data")   or {}).get("id")
            )

        if signal_id:
            open_signals[signal_id] = {
                "signal_id":   signal_id,
                "symbol":      symbol,
                "side":        side.lower(),
                "entry_price": levels["entry_price"],
                "stop_loss":   levels["stop_loss"],
                "take_profit": levels["take_profit"],
                "signal_time": row["timestamp"].isoformat(),
            }
            print(f"[OK] Señal almacenada: {symbol} {side} ID={signal_id}")
        else:
            print("[WARN] Señal creada pero sin signal_id en la respuesta.")

    except Exception as e:
        print(f"[ERROR] send_signal {symbol}: {e}")


# =========================
# BACKEND — CERRAR SEÑAL
# =========================
def close_signal(signal_id: str, close_price: float, reason: str):
    headers = {
        "Content-Type": "application/json",
        "x-bot-api-key": BOT_API_KEY,
    }
    payload = {
        "signal_id":      signal_id,
        "close_price":    close_price,
        "close_time":     pd.Timestamp.utcnow().isoformat(),
        "closing_reason": reason,
    }
    try:
        r = requests.post(
            f"{LOVABLE_BASE_URL}/bot-signals-close",
            json=payload, headers=headers, timeout=20
        )
        print(f"[CLOSE] {signal_id}: {r.status_code} | {reason}")

        if r.status_code in (200, 201) and signal_id in open_signals:
            del open_signals[signal_id]

    except Exception as e:
        print(f"[ERROR] close_signal {signal_id}: {e}")


# =========================
# MONITOR SEÑALES ABIERTAS
# =========================
def live_r(signal_data: dict, price: float) -> float:
    entry = signal_data["entry_price"]
    stop  = signal_data["stop_loss"]
    side  = signal_data["side"]
    risk  = abs(entry - stop)
    if risk <= 0:
        return 0.0
    return (price - entry) / risk if side == "long" else (entry - price) / risk


def monitor_open_signals():
    if not open_signals:
        return

    for signal_id in list(open_signals.keys()):
        sig = open_signals.get(signal_id)
        if not sig:
            continue

        price = get_current_price(sig["symbol"])
        if price is None:
            continue

        r_val = live_r(sig, price)
        print(f"[MONITOR] {sig['symbol']} {sig['side']} | price={price:.5f} | R={r_val:.2f}")

        side = sig["side"]
        tp   = sig["take_profit"]
        sl   = sig["stop_loss"]

        if side == "long":
            if price >= tp:
                close_signal(signal_id, price, "TP hit")
            elif price <= sl:
                close_signal(signal_id, price, "SL hit")
        else:
            if price <= tp:
                close_signal(signal_id, price, "TP hit")
            elif price >= sl:
                close_signal(signal_id, price, "SL hit")


# =========================
# PROCESAR SÍMBOLO
# =========================
def process_symbol(symbol: str):
    # Cooldown: no repetir señal en la misma vela
    side, row, strategy_name = get_confirmed_signal(symbol)
    if side is None or row is None:
        return

    candle_id    = str(row["timestamp"])
    candle_index = row.name if isinstance(row.name, int) else 0

    if last_alerted.get(symbol) == candle_id:
        return

    prev_idx = last_signal_idx.get(symbol)
    if prev_idx is not None and (candle_index - prev_idx) < COOLDOWN_CANDLES:
        return

    send_signal(symbol, side, strategy_name, row)
    last_alerted[symbol]    = candle_id
    last_signal_idx[symbol] = candle_index
    print(f"[SENT] {symbol} {side}")


# =========================
# MAIN LOOP
# =========================
def main():
    print("=" * 55)
    print("  EMA/MACD Trend Bot  |  3m + 1H  |  10 pares")
    print("=" * 55)
    print(f"Pares: {', '.join(TOP_10_SYMBOLS)}")
    print(f"Backend: {LOVABLE_BASE_URL}")
    print("=" * 55)

    while True:
        try:
            for symbol in TOP_10_SYMBOLS:
                try:
                    process_symbol(symbol)
                    time.sleep(0.5)   # respetar rate limit de Binance
                except Exception as e:
                    print(f"[ERROR] {symbol}: {e}")

            monitor_open_signals()

        except Exception as e:
            print(f"[ERROR] Loop general: {e}")

        print(f"Ciclo completo. Próximo en {CHECK_EVERY_SECONDS}s...\n")
        time.sleep(CHECK_EVERY_SECONDS)


if __name__ == "__main__":
    main()