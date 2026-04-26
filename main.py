# Импорт load_dotenv для загрузки переменных из файла .env


# Импорт load_dotenv для локальной загрузки .env (на хостинге bothost не используется)
from dotenv import load_dotenv
import os
if os.path.exists(".env"):
    load_dotenv()

import os
import logging
import requests
import pandas as pd
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import asyncio
import time
import websockets
from deriv_api import DerivAPI

# -------------------- Настройка логирования --------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# -------------------- Конфигурационные константы --------------------
# Токены и ключи из переменных окружения
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TWELVEDATA_KEY = os.environ.get("TWELVEDATA_KEY")
TD_BASE_URL = "https://api.twelvedata.com"

# Deriv настройки
DERIV_APP_ID = os.environ.get("DERIV_APP_ID")
DERIV_API_TOKEN = os.environ.get("DERIV_API_TOKEN")
DERIV_ACCOUNT_ID = os.environ.get("DERIV_ACCOUNT_ID")
DERIV_DEMO = os.environ.get("DERIV_DEMO", "True").lower() == "true"

# Список валютных и криптовалютных пар для мониторинга
MONITORED_PAIRS = [
    # Forex (налоговые пары, для них автотрейдинг не используется)
    "EUR/USD", "GBP/USD", "USD/JPY",
    # Криптовалюты (для них будет автотрейдинг на Deriv)
    "BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD",
    "ADA/USD", "SOL/USD", "DOGE/USD", "DOT/USD", "LTC/USD"
]

CHECK_INTERVAL_MINUTES = 60          # период проверки сигналов
SIGNAL_DURATION_MINUTES = 5          # длительность торговой позиции (контракта)
CANDLE_INTERVAL = "5min"             # таймфрейм свечей для анализа
OUTPUT_SIZE = 100                    # количество свечей
API_DELAY_SECONDS = 10               # задержка между запросами к Twelve Data (rate limit)

# Пороги профита/убытка для статистики
PROFIT_THRESHOLD_PCT = 0.1           # 0.1% для WIN
LOSS_THRESHOLD_PCT = 0.1             # 0.1% для LOSS

# Параметры индикатора CMA (Colored Moving Average)
CMA_LENGTH = 5                       # длина скользящей средней
CMA_TYPE = 1                         # тип: 1-Modified Hull, 2-Hull, 3-EMA, 4-WMA, 5-RMA, 6-VWMA, 7-SMA

last_signals = {}                    # {pair: последний сигнал} для предотвращения повторов
last_long_pos = {}                   # {pair: предыдущее состояние long (True/False)}
last_short_pos = {}                  # {pair: предыдущее состояние short (True/False)}
subscribed_chats = set()             # чаты, подписанные на автоматические сигналы
open_signals = []                    # активные сигналы (включая информацию об ордерах Deriv)
closed_signals = []                  # завершённые сигналы для статистики

# -------------------- Вспомогательные функции --------------------
def fetch_forex_data(from_currency: str, to_currency: str) -> pd.DataFrame:
    """Загрузка исторических данных OHLC с Twelve Data с задержкой для соблюдения лимита."""
    time.sleep(API_DELAY_SECONDS)
    symbol = f"{from_currency}/{to_currency}"
    params = {
        "symbol": symbol,
        "interval": CANDLE_INTERVAL,
        "apikey": TWELVEDATA_KEY,
        "outputsize": OUTPUT_SIZE
    }
    logger.info(f"Запрос данных для {symbol}")
    response = requests.get(f"{TD_BASE_URL}/time_series", params=params)
    response.raise_for_status()
    data = response.json()
    if "values" not in data:
        logger.error(f"Ошибка API: {data}")
        return pd.DataFrame()
    df = pd.DataFrame(data["values"]).iloc[::-1]
    df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"}, inplace=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    logger.info(f"Получено {len(df)} свечей для {symbol}")
    return df

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Расчёт индикатора RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def compute_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    """Расчёт полос Боллинджера."""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower})

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Расчёт индикатора MACD."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})

def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Расчёт стохастического осциллятора."""
    lowest_low = df["low"].rolling(window=k_period).min()
    highest_high = df["high"].rolling(window=k_period).max()
    k_percent = 100 * ((df["close"] - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return pd.DataFrame({"%K": k_percent, "%D": d_percent})

def find_support_resistance(df: pd.DataFrame):
    """Локальные уровни поддержки и сопротивления."""
    local_mins = df["low"][(df["low"].shift(1) > df["low"]) & (df["low"].shift(-1) > df["low"])]
    local_maxs = df["high"][(df["high"].shift(1) < df["high"]) & (df["high"].shift(-1) < df["high"])]
    current_price = df["close"].iloc[-1]
    supports = local_mins[local_mins < current_price]
    support = supports.max() if not supports.empty else np.nan
    resistances = local_maxs[local_maxs > current_price]
    resistance = resistances.min() if not resistances.empty else np.nan
    return support, resistance

def compute_cma(df: pd.DataFrame, length: int = 5, ma_type: int = 1) -> pd.Series:
    """
    Вычисление Colored Moving Average по спецификации скрипта.
    Типы:
      1 – Modified Hull (SMA of 2*WMA(len/2) - WMA(len))
      2 – Hull
      3 – EMA
      4 – WMA
      5 – RMA (Wilder's smoothing)
      6 – Volume Weighted (аппроксимировано WMA)
      7 – SMA
    """
    src = df["close"]
    half_length = max(1, length // 2)
    sqrt_length = max(1, int(np.round(np.sqrt(length))))

    if ma_type == 1:
        # Modified Hull
        wma_half = src.rolling(half_length).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
        wma_full = src.rolling(length).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
        hull_val = 2 * wma_half - wma_full
        return hull_val.rolling(sqrt_length).mean()
    elif ma_type == 2:
        # Hull
        wma_half = src.rolling(half_length).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
        wma_full = src.rolling(length).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
        hull_val = 2 * wma_half - wma_full
        return hull_val.rolling(sqrt_length).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
    elif ma_type == 3:
        return src.ewm(span=length, adjust=False).mean()
    elif ma_type == 4:
        return src.rolling(length).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
    elif ma_type == 5:
        return src.ewm(alpha=1/length, adjust=False).mean()
    elif ma_type == 6:
        # Volume Weighted – приближаем WMA
        return src.rolling(length).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
    elif ma_type == 7:
        return src.rolling(window=length).mean()
    else:
        return src.rolling(window=length).mean()

def analyze_pair_from_df(df: pd.DataFrame, from_currency: str, to_currency: str) -> dict:
    """
    Анализ на основе CMAs Close скрипта с дополнительными индикаторами.
    Возвращает словарь с сигналом и всеми значениями.
    """
    if df.empty or len(df) < 30:
        return {"error": "Недостаточно данных"}

    close_prices = df["close"]
    current_price = close_prices.iloc[-1]

    # Colored Moving Average
    cma_series = compute_cma(df, CMA_LENGTH, CMA_TYPE)
    cma_now = cma_series.iloc[-1]
    cma_prev = cma_series.iloc[-2] if len(cma_series) > 1 else cma_now
    cma_trend_up = cma_now > cma_prev
    cma_trend_color = "UP" if cma_trend_up else "DOWN"

    # Сигналы по правилам скрипта
    long_condition = current_price > cma_now and cma_trend_up
    short_condition = current_price < cma_now and not cma_trend_up
    if long_condition:
        signal = "BUY"
    elif short_condition:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Дополнительные индикаторы для отображения
    sma_fast = close_prices.rolling(window=10).mean().iloc[-1]
    sma_slow = close_prices.rolling(window=30).mean().iloc[-1]
    rsi = compute_rsi(close_prices, 14).iloc[-1]
    bb = compute_bollinger_bands(close_prices, 20, 2)
    bb_upper = bb["upper"].iloc[-1]
    bb_middle = bb["middle"].iloc[-1]
    bb_lower = bb["lower"].iloc[-1]
    macd = compute_macd(close_prices, 12, 26, 9)
    macd_line = macd["macd"].iloc[-1]
    macd_signal = macd["signal"].iloc[-1]
    macd_hist = macd["histogram"].iloc[-1]
    stoch = compute_stochastic(df, 14, 3)
    stoch_k = stoch["%K"].iloc[-1]
    stoch_d = stoch["%D"].iloc[-1]
    support, resistance = find_support_resistance(df)

    return {
        "pair": f"{from_currency}/{to_currency}",
        "price": round(current_price, 5),
        "cma": round(cma_now, 5),
        "cma_trend": cma_trend_color,
        "cma_type": CMA_TYPE,
        "cma_length": CMA_LENGTH,
        "sma_fast": round(sma_fast, 5),
        "sma_slow": round(sma_slow, 5),
        "rsi": round(rsi, 2),
        "bollinger_upper": round(bb_upper, 5),
        "bollinger_middle": round(bb_middle, 5),
        "bollinger_lower": round(bb_lower, 5),
        "macd": round(macd_line, 5),
        "macd_signal": round(macd_signal, 5),
        "macd_hist": round(macd_hist, 5),
        "stoch_k": round(stoch_k, 2),
        "stoch_d": round(stoch_d, 2),
        "support": round(support, 5) if not np.isnan(support) else None,
        "resistance": round(resistance, 5) if not np.isnan(resistance) else None,
        "signal": signal,
        "long_cond": long_condition,
        "short_cond": short_condition
    }

async def fetch_forex_data_async(from_currency: str, to_currency: str) -> pd.DataFrame:
    """Асинхронная версия загрузки данных (использует asyncio.sleep для совместимости с event loop)."""
    await asyncio.sleep(API_DELAY_SECONDS)
    symbol = f"{from_currency}/{to_currency}"
    params = {
        "symbol": symbol,
        "interval": CANDLE_INTERVAL,
        "apikey": TWELVEDATA_KEY,
        "outputsize": OUTPUT_SIZE
    }
    logger.info(f"Запрос данных для {symbol}")
    # requests – синхронный, но в фоновом потоке норм; для асинхронности можно использовать aiohttp, но оставим так
    response = await asyncio.get_event_loop().run_in_executor(None, lambda: requests.get(f"{TD_BASE_URL}/time_series", params=params))
    response.raise_for_status()
    data = response.json()
    if "values" not in data:
        logger.error(f"Ошибка API: {data}")
        return pd.DataFrame()
    df = pd.DataFrame(data["values"]).iloc[::-1]
    df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"}, inplace=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    logger.info(f"Получено {len(df)} свечей для {symbol}")
    return df

# -------------------- Вспомогательные функции для Deriv --------------------
def get_deriv_websocket_url(account_id: str, api_token: str, app_id: str, demo: bool = True) -> str:
    """
    Получает аутентифицированный WebSocket URL через REST API Deriv.
    """
    url = f"https://api.derivws.com/trading/v1/options/accounts/{account_id}/otp"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Deriv-App-ID": app_id,
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    if "websocket_url" in data:
        return data["websocket_url"]
    elif "url" in data:
        return data["url"]
    else:
        raise Exception(f"Не удалось получить WebSocket URL: {data}")

def to_deriv_symbol(pair: str) -> str:
    """
    Конвертирует пару 'BTC/USD' в 'cryBTCUSD' для Deriv.
    """
    base, quote = pair.split("/")
    return f"cry{base}{quote}"

def is_forex_pair(pair: str) -> bool:
    """
    Проверяет, является ли пара форекс-парой (для них автотрейдинг не используется).
    """
    forex_bases = ["EUR", "GBP", "USD", "JPY"]
    base = pair.split("/")[0]
    return base in forex_bases

async def deriv_place_order(api: DerivAPI, symbol: str, contract_type: str, amount: float) -> dict:
    """
    Размещает ордер на покупку контракта (CALL/PUT) на Deriv.
    """
    try:
        proposal = await api.proposal({
            "proposal": 1,
            "amount": amount,
            "basis": "stake",
            "contract_type": contract_type,  # "CALL" или "PUT"
            "currency": "USD",
            "duration": SIGNAL_DURATION_MINUTES,
            "duration_unit": "m",
            "symbol": symbol
        })

        if "error" in proposal:
            logger.error(f"Ошибка предложения Deriv: {proposal['error']}")
            return None

        buy = await api.buy({
            "buy": proposal["proposal"]["id"],
            "price": amount
        })
        logger.info(f"Deriv ордер {symbol} {contract_type} {amount}: {buy}")
        return buy
    except Exception as e:
        logger.error(f"Ошибка ордера Deriv: {e}")
        return None

async def deriv_close_position(api: DerivAPI, contract_id: str) -> dict:
    """
    Закрывает контракт на Deriv.
    """
    try:
        sell = await api.sell({
            "sell": contract_id,
            "price": 0  # Рыночная цена
        })
        logger.info(f"Deriv контракт {contract_id} закрыт: {sell}")
        return sell
    except Exception as e:
        logger.error(f"Ошибка закрытия контракта Deriv: {e}")
        return None

# -------------------- Асинхронное управление сигналами --------------------
async def add_signal_async(pair: str, signal_type: str, entry_price: float, api: DerivAPI = None) -> bool:
    """
    Добавляет новый сигнал и открывает позицию на Deriv (если передан API).
    """
    if any(s["pair"] == pair for s in open_signals):
        logger.info(f"Открытый сигнал по {pair} уже существует")
        return False

    symbol = None
    qty = None
    contract_id = None

    if api and not is_forex_pair(pair):
        symbol = to_deriv_symbol(pair)
        # Фиксированные суммы ставок
        if "BTC" in symbol:
            qty = 10.0
        elif "ETH" in symbol:
            qty = 10.0
        else:
            qty = 10.0  # Минимальная ставка
        contract_type = "CALL" if signal_type == "BUY" else "PUT"

        resp = await deriv_place_order(api, symbol, contract_type, qty)
        if resp and "buy" in resp:
            contract_id = resp["buy"].get("contract_id")
            logger.info(f"Реальная позиция открыта: {symbol} {contract_type} {qty} (контракт: {contract_id})")
        else:
            logger.warning("Не удалось открыть позицию Deriv, сигнал останется виртуальным")
            symbol = None
            qty = None

    open_signals.append({
        "pair": pair,
        "type": signal_type,
        "entry_price": entry_price,
        "timestamp": time.time(),
        "deriv_symbol": symbol,
        "deriv_qty": qty,
        "deriv_side": signal_type,
        "deriv_contract_id": contract_id
    })
    logger.info(f"Добавлен сигнал: {pair} {signal_type} по {entry_price}")
    return True

async def evaluate_open_signals_async(current_prices: dict, api: DerivAPI = None) -> None:
    """
    Закрывает истекшие сигналы и соответствующие контракты на Deriv.
    """
    global open_signals, closed_signals
    now = time.time()
    for sig in open_signals[:]:
        if now - sig["timestamp"] >= SIGNAL_DURATION_MINUTES * 60:
            pair = sig["pair"]
            current_price = current_prices.get(pair)
            if current_price is None:
                continue
            entry = sig["entry_price"]
            pct_change = (current_price - entry) / entry

            if sig["type"] == "BUY":
                if pct_change * 100 >= PROFIT_THRESHOLD_PCT:
                    result = "WIN"
                elif pct_change * 100 <= -LOSS_THRESHOLD_PCT:
                    result = "LOSS"
                else:
                    result = "DRAW"
            else:  # SELL
                if pct_change * 100 <= -PROFIT_THRESHOLD_PCT:
                    result = "WIN"
                elif pct_change * 100 >= LOSS_THRESHOLD_PCT:
                    result = "LOSS"
                else:
                    result = "DRAW"

            # Закрываем контракт на Deriv, если он был открыт
            if api and sig.get("deriv_contract_id"):
                await deriv_close_position(api, sig["deriv_contract_id"])

            closed_signals.append({
                "pair": pair,
                "type": sig["type"],
                "entry_price": entry,
                "exit_price": current_price,
                "result": result,
                "open_time": sig["timestamp"],
                "close_time": now
            })
            logger.info(f"Сигнал {pair} {sig['type']} закрыт: {result} (вход={entry}, выход={current_price}, изменение={pct_change*100:.2f}%)")
            open_signals.remove(sig)

def get_stats() -> tuple:
    """Возвращает (wins, losses, draws, total) по закрытым сигналам."""
    wins = sum(1 for s in closed_signals if s["result"] == "WIN")
    losses = sum(1 for s in closed_signals if s["result"] == "LOSS")
    draws = sum(1 for s in closed_signals if s["result"] == "DRAW")
    total = len(closed_signals)
    return wins, losses, draws, total

# -------------------- Обработчики команд Telegram --------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Подписка на автоматические сигналы."""
    chat_id = update.effective_chat.id
    subscribed_chats.add(chat_id)
    await update.message.reply_text(
        "🚀 Привет! Вы подписаны на сигналы CMAs Close (Whvntr/TradeStation).\n"
        f"Параметры: тип MA {CMA_TYPE}, длина {CMA_LENGTH}.\n"
        "Сигналы: BUY при close > MA и MA растёт; SELL при close < MA и MA падает.\n"
        "Автотрейдинг Deriv включён для криптовалют (CALL/PUT контракты).\n"
        f"Время контракта: {SIGNAL_DURATION_MINUTES} мин.\n"
        "Команды: /signal BTC/USD, /stats, /stop"
    )

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отписка от сигналов."""
    chat_id = update.effective_chat.id
    subscribed_chats.discard(chat_id)
    await update.message.reply_text("Вы отписались от сигналов.")

async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ручной запрос анализа и открытие позиции на Deriv при возможности."""
    if not context.args or len(context.args) != 1:
        await update.message.reply_text("Использование: /signal BTC/USD")
        return
    pair_input = context.args[0].upper().replace(" ", "")
    parts = pair_input.split("/")
    if len(parts) != 2:
        await update.message.reply_text("Неверный формат. Пример: BTC/USD")
        return
    base, quote = parts
    msg = await update.message.reply_text("Анализирую по CMAs...")
    # Загружаем данные (синхронно, но в асинхронном обработчике можно использовать fetch_forex_data,
    # т.к. он вызывает time.sleep, что блокирует event loop; лучше использовать асинхронную версию)
    df = await fetch_forex_data_async(base, quote)
    result = analyze_pair_from_df(df, base, quote)
    if "error" in result:
        await msg.edit_text(f"Ошибка: {result['error']}")
        return

    deriv_api = context.application.deriv_api
    if result["signal"] in ("BUY", "SELL"):
        await add_signal_async(result["pair"], result["signal"], result["price"], deriv_api)

    signal_emoji = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "⚪️ HOLD"}
    support_str = f"{result['support']}" if result['support'] else "N/A"
    resistance_str = f"{result['resistance']}" if result['resistance'] else "N/A"
    text = (
        f"📊 **{result['pair']}**\n"
        f"Цена: {result['price']}\n"
        f"CMA ({result['cma_length']}/{['ModHull','Hull','EMA','WMA','RMA','VWMA','SMA'][result['cma_type']-1]}): {result['cma']} ({result['cma_trend']})\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📈 SMA(10): {result['sma_fast']} | SMA(30): {result['sma_slow']}\n"
        f"📉 RSI(14): {result['rsi']}\n"
        f"🧿 Bollinger (20,2): U:{result['bollinger_upper']} M:{result['bollinger_middle']} L:{result['bollinger_lower']}\n"
        f"📊 MACD: Line={result['macd']} Signal={result['macd_signal']} Hist={result['macd_hist']}\n"
        f"🔄 Stochastic: %K={result['stoch_k']} %D={result['stoch_d']}\n"
        f"📌 Support: {support_str} | Resistance: {resistance_str}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"⭐ Сигнал (CMAs): {signal_emoji.get(result['signal'], result['signal'])}\n"
        f"Контракт на {SIGNAL_DURATION_MINUTES} мин."
    )
    await msg.edit_text(text, parse_mode="Markdown")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Статистика закрытых сигналов."""
    wins, losses, draws, total = get_stats()
    if total == 0:
        await update.message.reply_text("Статистика пока пуста.")
        return
    win_rate = (wins / total) * 100
    text = (
        f"📊 **Статистика сигналов**\n"
        f"Всего завершено: {total}\n"
        f"✅ WIN: {wins}\n"
        f"❌ LOSS: {losses}\n"
        f"➖ DRAW: {draws}\n"
        f"🏆 Процент побед: {win_rate:.1f}%\n"
        f"Порог: {PROFIT_THRESHOLD_PCT}% / {LOSS_THRESHOLD_PCT}%\n"
        f"Время контракта: {SIGNAL_DURATION_MINUTES} мин."
    )
    await update.message.reply_text(text, parse_mode="Markdown")

# -------------------- Задача автоматической рассылки сигналов --------------------
async def periodic_signal_check(app: Application) -> None:
    """Фоновая задача: анализ пар, генерация сигналов, работа с Deriv."""
    global last_long_pos, last_short_pos, last_signals
    await asyncio.sleep(10)  # задержка для инициализации бота
    logger.info("Запуск фоновой проверки сигналов (CMAs Close + Deriv)")
    deriv_api = app.deriv_api

    try:
        while True:
            logger.info("Проверка сигналов...")
            current_prices = {}
            for pair_str in MONITORED_PAIRS:
                pair_clean = pair_str.replace(" ", "").upper()
                parts = pair_clean.split("/")
                if len(parts) != 2:
                    continue
                base, quote = parts
                df = await fetch_forex_data_async(base, quote)
                if df.empty:
                    continue
                current_prices[pair_clean] = df["close"].iloc[-1]
                result = analyze_pair_from_df(df, base, quote)
                if "error" in result:
                    logger.error(f"Ошибка {pair_clean}: {result['error']}")
                    continue

                long_now = result["long_cond"]
                short_now = result["short_cond"]
                prev_long = last_long_pos.get(pair_clean, False)
                prev_short = last_short_pos.get(pair_clean, False)
                new_long = long_now and not prev_long
                new_short = short_now and not prev_short

                last_long_pos[pair_clean] = long_now
                last_short_pos[pair_clean] = short_now

                signal = None
                if new_long:
                    signal = "BUY"
                elif new_short:
                    signal = "SELL"

                if signal:
                    if pair_clean in last_signals and last_signals[pair_clean] == signal:
                        continue
                    last_signals[pair_clean] = signal

                    await add_signal_async(result["pair"], signal, result["price"], deriv_api)

                    signal_emoji = {"BUY": "🟢 BUY", "SELL": "🔴 SELL"}
                    support_str = f"{result['support']}" if result['support'] else "N/A"
                    resistance_str = f"{result['resistance']}" if result['resistance'] else "N/A"
                    text = (
                        f"🚨 **CMAs Сигнал ({result['pair']})**\n"
                        f"{signal_emoji[signal]} (цена: {result['price']})\n"
                        f"CMA {result['cma_length']}/{['ModHull','Hull','EMA','WMA','RMA','VWMA','SMA'][result['cma_type']-1]}: {result['cma']} ({result['cma_trend']})\n"
                        f"RSI: {result['rsi']} | Stochastic: %K={result['stoch_k']}\n"
                        f"Поддержка: {support_str} | Сопротивление: {resistance_str}\n"
                        f"Длительность контракта: {SIGNAL_DURATION_MINUTES} мин."
                    )
                    for chat_id in subscribed_chats.copy():
                        try:
                            await app.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
                        except Exception as e:
                            logger.error(f"Ошибка отправки в {chat_id}: {e}")
                            subscribed_chats.discard(chat_id)
                elif pair_clean in last_signals:
                    del last_signals[pair_clean]

            await evaluate_open_signals_async(current_prices, deriv_api)
            await asyncio.sleep(CHECK_INTERVAL_MINUTES * 60)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Фоновая задача остановлена")

# -------------------- Точка входа --------------------
def main() -> None:
    """Запуск Telegram-бота с автотрейдингом Deriv."""
    if not TELEGRAM_TOKEN or not TWELVEDATA_KEY:
        logger.critical("Установите TELEGRAM_TOKEN и TWELVEDATA_KEY в .env")
        return

    # Инициализация клиента Deriv
    deriv_api = None
    if DERIV_API_TOKEN and DERIV_APP_ID and DERIV_ACCOUNT_ID:
        try:
            ws_url = get_deriv_websocket_url(DERIV_ACCOUNT_ID, DERIV_API_TOKEN, DERIV_APP_ID, DERIV_DEMO)
            logger.info(f"Deriv WebSocket URL получен")
            deriv_api = DerivAPI(endpoint=ws_url, app_id=DERIV_APP_ID)
            logger.info("Deriv API инициализирован (демо=%s)", DERIV_DEMO)
        except Exception as e:
            logger.error(f"Ошибка инициализации Deriv API: {e}")
    else:
        logger.warning("Deriv ключи не заданы – автотрейдинг отключён")

    # Создание приложения Telegram
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    # Сохраняем клиент Deriv в приложении для доступа из обработчиков
    application.deriv_api = deriv_api

    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("signal", signal_command))
    application.add_handler(CommandHandler("stats", stats_command))

    # Запуск фоновой задачи
    loop = asyncio.get_event_loop()
    loop.create_task(periodic_signal_check(application))
    logger.info("Бот запущен (CMAs Close + Deriv автотрейдинг).")
    application.run_polling()

if __name__ == "__main__":
    main()
