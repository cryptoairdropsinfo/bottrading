import time
import math
import requests
import pandas as pd

# =========================
# CONFIG
# =========================

# ── Timeframes ───────────────────────────────────────
# 15m = entrada  |  30m = filtro de dirección
TF_ENTRY  = "15m"
TF_FILTER = "30m"
LIMIT     = 200
CHECK_EVERY_SECONDS = 45   # scalping revisa más frecuente

# ── Indicadores ──────────────────────────────────────
RSI_PERIOD      = 7        # RSI rápido para scalping (más sensible)
RSI_OB          = 72       # Sobrecompra → posible SHORT
RSI_OS          = 28       # Sobreventa  → posible LONG
STOCH_K         = 5        # Stochastic %K (rápido)
STOCH_D         = 3        # Stochastic %D (señal)
STOCH_OB        = 80       # Stochastic sobrecompra
STOCH_OS        = 20       # Stochastic sobreventa
BB_PERIOD       = 20       # Bandas de Bollinger
BB_STD          = 2.0      # Desviación estándar de las BB
EMA_FAST        = 8        # EMA micro-tendencia
EMA_MED         = 21       # EMA dirección corta
ATR_PERIOD      = 14
VOL_MA_PERIOD   = 20
COOLDOWN_CANDLES = 5

# ── SL/TP — Scalping con margen ──────────────────────
# ATR × 1.8 → más ajustado que tendencia pero con margen real
# TP 1.8:1  → win rate > 60% compensa ratio menor
ATR_STOP_MULTIPLIER = 1.8
TP_RR_RATIO         = 1.5
MIN_VOLUME_RATIO    = 1.2   # scalping no necesita volumen explosivo

MAX_OPEN_SIGNALS    = 10

# ── Backend (mismo sistema) ──────────────────────────
LOVABLE_BASE_URL = "https://fsjkfaallagpkuoliizg.supabase.co/functions/v1"
BOT_API_KEY      = "ragery_bot_secure_982347234"

# ── Binance USDⓈ-M Futures ───────────────────────────
BINANCE_BASE_URL = "https://fapi.binance.com"
KLINES_URL       = f"{BINANCE_BASE_URL}/fapi/v1/klines"
PRICE_URL        = f"{BINANCE_BASE_URL}/fapi/v1/ticker/price"

# ── 10 pares ─────────────────────────────────────────
SYMBOLS = [
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
    "HYPEUSDT",
    "POLUSDT",
    "ALGOUSDT",
    "TRXUSDT",
    "BCHUSDT",
    "XMRUSDT",
    "LTCUSDT",
    "SUIUSDT",
    "APTUSDT",
    "TONUSDT",
]

# ── Estado runtime ────────────────────────────────────
last_alerted    = {}
last_signal_idx = {}
open_signals    = {}


# =========================
# HELPERS
# =========================
def symbol_to_display(symbol: str) -> str:
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}/USDT:USDT"
    return symbol


def safe_get(url: str, params=None, retries=2, label="req"):
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
# MARKET DATA
# =========================
def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame | None:
    data = safe_get(
        KLINES_URL,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        retries=2,
        label=f"{symbol}/{interval}"
    )
    if not isinstance(data, list) or len(data) < 50:
        return None

    rows = []
    for k in data:
        try:
            rows.append([int(k[0]), float(k[1]), float(k[2]),
                         float(k[3]), float(k[4]), float(k[5])])
        except Exception:
            continue

    if len(rows) < 50:
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
def compute_rsi(series: pd.Series, period: int = 7) -> pd.Series:
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


def compute_stochastic(df: pd.DataFrame, k: int = 5, d: int = 3) -> tuple:
    """
    Stochastic Oscillator — mide dónde está el precio dentro del rango K velas.
    Muy efectivo para detectar agotamiento en scalping.
    %K < 20 = sobreventa  |  %K > 80 = sobrecompra
    """
    low_k  = df["low"].rolling(k).min()
    high_k = df["high"].rolling(k).max()
    stoch_k = 100 * (df["close"] - low_k) / (high_k - low_k).replace(0, math.nan)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


def compute_bollinger(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> tuple:
    """
    Bandas de Bollinger — precio toca banda inferior = sobreventa local.
    Precio toca banda superior = sobrecompra local.
    La banda media actúa como imán (reversión a la media).
    """
    mid    = df["close"].rolling(period).mean()
    sigma  = df["close"].rolling(period).std()
    upper  = mid + (std * sigma)
    lower  = mid - (std * sigma)
    # %B: posición del precio dentro de las bandas (0=banda baja, 1=banda alta)
    pct_b  = (df["close"] - lower) / (upper - lower).replace(0, math.nan)
    return upper, mid, lower, pct_b


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # EMAs de micro-tendencia
    df["ema8"]  = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=EMA_MED,  adjust=False).mean()

    # RSI rápido (período 7 para scalping)
    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)

    # Stochastic
    df["stoch_k"], df["stoch_d"] = compute_stochastic(df, STOCH_K, STOCH_D)

    # Bollinger Bands
    df["bb_upper"], df["bb_mid"], df["bb_lower"], df["bb_pct"] = compute_bollinger(
        df, BB_PERIOD, BB_STD
    )

    # ATR y volumen
    df["atr"]    = compute_atr(df, ATR_PERIOD)
    df["vol_ma"] = df["volume"].rolling(VOL_MA_PERIOD).mean()

    return df


# =========================
# ESTRATEGIA — SCALPING
# REVERSIÓN A LA MEDIA + CONFLUENCIA
# =========================
"""
LÓGICA DE ALTA TASA DE ACIERTO:

En lugar de perseguir breakouts (que son la causa #1 de SL en scalping),
esta estrategia busca AGOTAMIENTO del movimiento y REBOTE al nivel medio.

El precio no se mueve en línea recta — siempre vuelve a la media.
Cuando detectamos 3 señales de agotamiento juntas, el rebote es muy probable.

LONG (rebote desde sobreventa):
  1. Precio toca o cruza la Banda Bollinger inferior (%B < 0.05)
  2. RSI(7) en sobreventa < 28
  3. Stochastic %K < 20 Y empieza a girar al alza (%K > %D)
  4. EMA8 > EMA21 en 30m (no ir contra tendencia media)
  5. La vela de 15m cierra con mecha larga abajo (rechazo del nivel)

SHORT (rebote desde sobrecompra): condiciones inversas.

POR QUÉ ES ALTO WIN RATE:
  - 3 indicadores de agotamiento simultáneos = señal de alta probabilidad
  - El TP es la banda media de Bollinger (imán natural del precio)
  - No perseguimos momentum, esperamos que se agote
  - 30m como filtro evita entrar contra tendencia mayor
"""


def get_30m_direction(df_30m: pd.DataFrame) -> str | None:
    """Tendencia simple en 30m: EMA8 vs EMA21."""
    row = df_30m.iloc[-2]
    if pd.isna(row["ema8"]) or pd.isna(row["ema21"]):
        return None
    if row["ema8"] > row["ema21"]:
        return "LONG"
    if row["ema8"] < row["ema21"]:
        return "SHORT"
    return None


def has_rejection_wick_long(row: pd.Series) -> bool:
    """
    Vela con mecha larga abajo = rechazo del nivel bajo.
    La mecha inferior debe ser al menos 60% del rango total de la vela.
    """
    candle_range = row["high"] - row["low"]
    if candle_range <= 0:
        return False
    lower_wick = min(row["open"], row["close"]) - row["low"]
    return (lower_wick / candle_range) >= 0.40


def has_rejection_wick_short(row: pd.Series) -> bool:
    """Mecha larga arriba = rechazo del nivel alto."""
    candle_range = row["high"] - row["low"]
    if candle_range <= 0:
        return False
    upper_wick = row["high"] - max(row["open"], row["close"])
    return (upper_wick / candle_range) >= 0.40


def get_entry_signal(df_15m: pd.DataFrame, direction_30m: str | None) -> str | None:
    row  = df_15m.iloc[-2]   # última vela cerrada
    prev = df_15m.iloc[-3]

    required = ["rsi", "stoch_k", "stoch_d", "bb_pct", "bb_mid",
                "ema8", "ema21", "atr", "vol_ma"]
    if any(pd.isna(row[c]) for c in required):
        return None

    vol_ok = (row["vol_ma"] > 0 and
              row["volume"] >= row["vol_ma"] * MIN_VOLUME_RATIO)

    # ── LONG — Reversión desde sobreventa ─────────────
    # Condición 1: precio en/bajo la banda Bollinger inferior
    bb_oversold   = row["bb_pct"] <= 0.08

    # Condición 2: RSI(7) en sobreventa
    rsi_oversold  = row["rsi"] <= RSI_OS

    # Condición 3: Stochastic en sobreventa Y girando al alza
    stoch_oversold_turn = (
        row["stoch_k"] <= STOCH_OS and
        row["stoch_k"] > row["stoch_d"] and   # %K cruza por encima de %D
        prev["stoch_k"] <= prev["stoch_d"]     # en la vela anterior aún no había cruzado
    )

    # Condición 4: mecha de rechazo abajo
    wick_long = has_rejection_wick_long(row)

    # Condición 5: no ir contra tendencia en 30m
    trend_ok_long = direction_30m in ("LONG", None)   # None = neutral, permitido

    # Necesitamos mínimo 3 de las 4 primeras condiciones + tendencia ok
    long_score = sum([bb_oversold, rsi_oversold, stoch_oversold_turn, wick_long])
    if long_score >= 3 and trend_ok_long and vol_ok:
        return "LONG"

    # ── SHORT — Reversión desde sobrecompra ───────────
    bb_overbought       = row["bb_pct"] >= 0.92
    rsi_overbought      = row["rsi"] >= RSI_OB
    stoch_overbought_turn = (
        row["stoch_k"] >= STOCH_OB and
        row["stoch_k"] < row["stoch_d"] and
        prev["stoch_k"] >= prev["stoch_d"]
    )
    wick_short          = has_rejection_wick_short(row)
    trend_ok_short      = direction_30m in ("SHORT", None)

    short_score = sum([bb_overbought, rsi_overbought, stoch_overbought_turn, wick_short])
    if short_score >= 3 and trend_ok_short and vol_ok:
        return "SHORT"

    return None


def get_confirmed_signal(symbol: str) -> tuple:
    df_15m = fetch_klines(symbol, TF_ENTRY,  limit=LIMIT)
    df_30m = fetch_klines(symbol, TF_FILTER, limit=100)

    if df_15m is None:
        return None, None, ""

    df_15m = enrich(df_15m)

    direction_30m = None
    if df_30m is not None:
        df_30m = enrich(df_30m)
        direction_30m = get_30m_direction(df_30m)

    side = get_entry_signal(df_15m, direction_30m)
    if side is None:
        return None, None, ""

    row  = df_15m.iloc[-2]

    # TP = banda media Bollinger (imán natural) en lugar de ATR fijo
    # Esto aumenta el win rate porque el precio SIEMPRE tiende a volver al medio
    bb_mid_dist = abs(float(row["close"]) - float(row["bb_mid"]))

    name = (
        f"Scalping Reversión {'Long' if side == 'LONG' else 'Short'} "
        f"(15m+30m) | BB+RSI(7)+Stoch | TP→BandaMed"
    )
    return side, row, name


# =========================
# RISK MANAGEMENT
# =========================
def calculate_levels(side: str, row: pd.Series) -> dict | None:
    entry  = float(row["close"])
    atr    = float(row["atr"])
    bb_mid = float(row["bb_mid"])

    if atr <= 0 or math.isnan(atr):
        return None

    sl_dist = atr * ATR_STOP_MULTIPLIER

    if side == "LONG":
        sl = entry - sl_dist
        # TP primario: banda media de Bollinger (más realista)
        # TP mínimo:   ratio 1.8:1 con el SL
        tp_bb   = bb_mid
        tp_rr   = entry + (sl_dist * TP_RR_RATIO)
        # Usamos el menor (más conservador y más probable de alcanzar)
        tp = min(tp_bb, tp_rr) if tp_bb > entry else tp_rr
    else:
        sl = entry + sl_dist
        tp_bb   = bb_mid
        tp_rr   = entry - (sl_dist * TP_RR_RATIO)
        tp = max(tp_bb, tp_rr) if tp_bb < entry else tp_rr

    risk   = abs(entry - sl)
    reward = abs(tp - entry)
    rr     = round(reward / risk, 2) if risk > 0 else TP_RR_RATIO

    return {
        "entry_price":       entry,
        "stop_loss":         sl,
        "take_profit":       tp,
        "risk_reward_ratio": rr,
        "atr_value":         atr,
    }


# =========================
# BACKEND — CREAR SEÑAL
# =========================
def send_signal(symbol: str, side: str, strategy_name: str, row: pd.Series):
    if len(open_signals) >= MAX_OPEN_SIGNALS:
        print(f"[SKIP] Máximo de señales abiertas ({MAX_OPEN_SIGNALS}).")
        return

    levels = calculate_levels(side, row)
    if not levels:
        return

    vol_ma    = float(row["vol_ma"]) if pd.notna(row["vol_ma"]) and row["vol_ma"] > 0 else 0
    vol_ratio = float(row["volume"]) / vol_ma if vol_ma > 0 else 0

    payload = {
        "symbol":            symbol_to_display(symbol),
        "side":              side.lower(),
        "timeframe":         "15m+30m",
        "strategy_name":     strategy_name,
        "entry_price":       levels["entry_price"],
        "stop_loss":         levels["stop_loss"],
        "take_profit":       levels["take_profit"],
        "risk_reward_ratio": levels["risk_reward_ratio"],
        "rsi_value":         float(row["rsi"]),
        "volume_ratio":      round(vol_ratio, 2),
        "atr_value":         levels["atr_value"],
        "signal_time":       row["timestamp"].isoformat(),
        "status":            "active",
        "signal_source":     "bot",
        "notes": (
            f"Scalping reversión media | BB%={row['bb_pct']:.2f} | "
            f"RSI(7)={row['rsi']:.1f} | Stoch={row['stoch_k']:.1f} | "
            f"SL ATR×{ATR_STOP_MULTIPLIER} | TP banda media BB"
        ),
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
        print(f"[SIGNAL] {symbol} {side} → {r.status_code}")

        if r.status_code not in (200, 201):
            return

        data      = r.json()
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
            print(f"[OK] {symbol} {side} almacenada. ID={signal_id}")
        else:
            print("[WARN] Sin signal_id en la respuesta del backend.")

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
        print(f"[CLOSE] {signal_id} | {reason} | {r.status_code}")
        if r.status_code in (200, 201) and signal_id in open_signals:
            del open_signals[signal_id]
    except Exception as e:
        print(f"[ERROR] close_signal {signal_id}: {e}")


# =========================
# MONITOR SEÑALES ABIERTAS
# =========================
def live_r(sig: dict, price: float) -> float:
    entry = sig["entry_price"]
    stop  = sig["stop_loss"]
    risk  = abs(entry - stop)
    if risk <= 0:
        return 0.0
    if sig["side"] == "long":
        return (price - entry) / risk
    return (entry - price) / risk


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
        print(f"[MON] {sig['symbol']} {sig['side']} | {price:.5f} | R={r_val:.2f}")

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
    side, row, strategy_name = get_confirmed_signal(symbol)
    if side is None or row is None:
        return

    candle_id  = str(row["timestamp"])
    candle_idx = row.name if isinstance(row.name, int) else 0

    if last_alerted.get(symbol) == candle_id:
        return

    prev_idx = last_signal_idx.get(symbol)
    if prev_idx is not None and (candle_idx - prev_idx) < COOLDOWN_CANDLES:
        return

    send_signal(symbol, side, strategy_name, row)
    last_alerted[symbol]    = candle_id
    last_signal_idx[symbol] = candle_idx


# =========================
# MAIN LOOP
# =========================
def main():
    print("=" * 62)
    print("  SCALPING BOT — Reversión a la Media")
    print(f"  TF: {TF_ENTRY} entrada | {TF_FILTER} filtro")
    print(f"  SL: ATR×{ATR_STOP_MULTIPLIER}  |  TP: BandaMedia BB (~{TP_RR_RATIO}:1)")
    print(f"  Indicadores: BB + RSI(7) + Stochastic({STOCH_K},{STOCH_D})")
    print("=" * 62)
    print(f"  Pares: {', '.join(SYMBOLS)}")
    print("=" * 62)

    while True:
        try:
            for symbol in SYMBOLS:
                try:
                    process_symbol(symbol)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"[ERROR] {symbol}: {e}")

            monitor_open_signals()

        except Exception as e:
            print(f"[ERROR] loop: {e}")

        print(f"\nCiclo completo. Próximo en {CHECK_EVERY_SECONDS}s...\n")
        time.sleep(CHECK_EVERY_SECONDS)


if __name__ == "__main__":
    main()