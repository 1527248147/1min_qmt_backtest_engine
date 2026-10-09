#coding:gbk
"""
QMT model-backtest strategy for the main ML stock-selection signal.

Run this script inside QMT's built-in Python model backtest.  Use a 1m period
if you want the 14:56 / 14:57 close-sell behavior to be exercised by QMT's
historical model engine.  The script does not use run_time.
"""
import csv
import datetime as dt


PLAN_CSV = (
    "C:\\AI_STOCK\\\u7814\u62a5\\Machine Learning for Stock Selection\\"
    "qmt_backtest\\qmt_targets_top10.csv"
)

ACCOUNT_ID = "testS"
FALLBACK_CAPITAL = 1000000

PRICE_STYLE = "COMPETE"
SELL_STYLE_COMPETE = 14
SELL_STYLE_LIMIT_DOWN = 41

BUY_TIME = "093000"
SELL_START_TIME = "145600"
CLOSING_AUCTION_TIME = "145700"


def _account_total_value(C):
    try:
        rows = get_trade_detail_data(C.account_id, "STOCK", "ACCOUNT")
    except Exception:
        try:
            rows = get_trade_detail_data(C.account_id, "stock", "account")
        except Exception:
            return float(getattr(C, "capital", 0) or getattr(C, "asset", 0) or FALLBACK_CAPITAL)

    for obj in rows:
        value = getattr(obj, "m_dBalance", None)
        if value and value > 0:
            return float(value)
    return float(getattr(C, "capital", 0) or getattr(C, "asset", 0) or FALLBACK_CAPITAL)


def _debug_qmt_market_data(C, trade_date):
    if trade_date in C.debug_market_dates:
        return
    C.debug_market_dates.add(trade_date)
    codes = ["600738.SH", "603567.SH", "603600.SH"]
    try:
        data = get_market_data_ex(
            fields=["open", "close", "volume", "amount"],
            stock_code=codes,
            period="1m",
            start_time=trade_date + "093000",
            end_time=trade_date + "093200",
            dividend_type=getattr(C, "dividend_type", ""),
            fill_data=False,
        )
        for code in codes:
            frame = data.get(code)
            print("QMT DATA CHECK", trade_date, code, frame)
    except Exception as e:
        print("QMT DATA CHECK failed", trade_date, repr(e))


def _load_plan(path):
    by_signal_date = {}
    by_trade_date = {}
    all_codes = set()

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            signal_date = row["signal_date"]
            trade_date = row["trade_date"]
            code = row["ts_code"]
            rank = int(row["rank"])
            weight = float(row["target_weight"])
            target_slots = int(row["target_slots"])
            item = (rank, code, weight, target_slots, signal_date, trade_date)
            by_signal_date.setdefault(signal_date, []).append(item)
            by_trade_date.setdefault(trade_date, []).append(item)
            all_codes.add(code)

    for d in by_signal_date:
        by_signal_date[d].sort()
    for d in by_trade_date:
        by_trade_date[d].sort()
    return by_signal_date, by_trade_date, all_codes


def _bar_datetime(C):
    timetag = C.get_bar_timetag(C.barpos)
    if timetag:
        china_time = dt.datetime.utcfromtimestamp(timetag / 1000.0) + dt.timedelta(hours=8)
        return china_time.strftime("%Y%m%d"), china_time.strftime("%H%M%S")

    period = str(getattr(C, "period", "")).lower()
    if period in ("1d", "d", "day") or period.endswith("d"):
        day = timetag_to_datetime(timetag, "%Y%m%d")
        return day, "150000"

    try:
        stamp = timetag_to_datetime(timetag, "%Y%m%d%H%M%S")
        if len(stamp) >= 14:
            return stamp[:8], stamp[8:14]
    except Exception:
        pass

    day = timetag_to_datetime(timetag, "%Y%m%d")
    return day, "150000"


def _positions(C, use_available=True):
    try:
        rows = get_trade_detail_data(C.account_id, "STOCK", "POSITION")
    except Exception:
        try:
            rows = get_trade_detail_data(C.account_id, "stock", "position")
        except Exception as e:
            print("get positions failed:", e)
            return {}

    out = {}
    for obj in rows:
        code = getattr(obj, "m_strInstrumentID", "")
        market = getattr(obj, "m_strExchangeID", "")
        if use_available:
            volume = getattr(
                obj,
                "m_nCanUseVolume",
                getattr(obj, "can_use_volume", getattr(obj, "m_nVolume", getattr(obj, "volume", 0))),
            )
        else:
            volume = getattr(obj, "m_nVolume", getattr(obj, "volume", 0))
        if code and market and volume:
            out[code + "." + market] = int(volume)
    return out


def _held_codes(C):
    return set(_positions(C, use_available=False))


def _sync_pending_sells(C):
    if not C.pending_sells:
        return
    held = _held_codes(C)
    cleared = C.pending_sells - held
    if cleared:
        print("pending sells cleared", len(cleared), sorted(cleared)[:8])
    C.pending_sells = C.pending_sells & held


def _mark_stale_for_signal_date(C, signal_date):
    if signal_date in C.stale_marked_dates:
        return
    rows = C.plan_by_signal_date.get(signal_date)
    if not rows:
        return

    target_codes = set(item[1] for item in rows)
    held = _held_codes(C)
    stale = held - target_codes
    if stale:
        C.pending_sells.update(stale)
        print("mark stale", signal_date, "count", len(stale), sorted(stale)[:8])
    C.stale_marked_dates.add(signal_date)


def _sell_pending(C, prtype, tag):
    _sync_pending_sells(C)
    if not C.pending_sells:
        return

    available = _positions(C, use_available=True)
    sent = 0
    for code in sorted(C.pending_sells):
        shares = available.get(code, 0)
        if shares <= 0:
            continue
        remark = tag + "_" + code.replace(".", "_")
        try:
            passorder(24, 1101, C.account_id, code, prtype, -1, shares, tag, 2, remark, C)
        except TypeError:
            passorder(24, 1101, C.account_id, code, prtype, -1, shares, C)
        sent += 1
    if sent:
        print("sell pending", tag, "orders", sent, "prtype", prtype)


def _buy_targets(C, trade_date):
    if trade_date in C.buy_done_dates:
        return
    rows = C.plan_by_trade_date.get(trade_date)
    if not rows:
        return

    _debug_qmt_market_data(C, trade_date)
    _sync_pending_sells(C)
    held = _held_codes(C)
    target_codes = set(item[1] for item in rows)
    target_slots = rows[0][3]

    pending_held = C.pending_sells & held
    retained = held & target_codes
    open_slots = max(0, target_slots - len(pending_held) - len(retained))

    desired = set(retained)
    for _, code, _, _, _, _ in rows:
        if len(desired) >= len(retained) + open_slots:
            break
        desired.add(code)

    weights = {code: weight for _, code, weight, _, _, _ in rows}
    total_value = _account_total_value(C)
    order_errors = 0
    sent = 0
    for code in sorted(desired):
        target_value = total_value * weights[code]
        try:
            order_target_value(code, target_value, PRICE_STYLE, C, C.account_id)
            sent += 1
        except Exception as e:
            order_errors += 1
            if order_errors <= 10:
                print("buy order error", trade_date, code, repr(e))

    C.buy_done_dates.add(trade_date)
    print(
        "buy rebalance",
        trade_date,
        "desired",
        len(desired),
        "target_slots",
        target_slots,
        "total_value",
        total_value,
        "per_stock_value",
        total_value / target_slots if target_slots else 0,
        "orders_sent",
        sent,
        "pending_sells",
        len(pending_held),
        "order_errors",
        order_errors,
    )


def init(C):
    C.account_id = ACCOUNT_ID
    C.plan_by_signal_date, C.plan_by_trade_date, C.plan_codes = _load_plan(PLAN_CSV)

    C.pending_sells = set()
    C.stale_marked_dates = set()
    C.close_sell_done = set()
    C.auction_sell_done = set()
    C.buy_done_dates = set()
    C.debug_bar_count = 0
    C.debug_seen_plan_dates = set()
    C.debug_seen_days = set()
    C.debug_market_dates = set()

    signal_dates = sorted(C.plan_by_signal_date)
    trade_dates = sorted(C.plan_by_trade_date)
    if trade_dates:
        C.start = signal_dates[0] + " 00:00:00"
        C.end = trade_dates[-1] + " 15:30:00"
        C.start_time = signal_dates[0]
        C.end_time = trade_dates[-1]

    # Do not push all historical A-share symbols into set_universe.  QMT's
    # current instrument master rejects some delisted/renamed historical codes.
    # Orders below still use their explicit stock codes from the plan.
    try:
        C.set_universe(["000001.SZ"])
    except Exception as e:
        print("set benchmark universe failed:", e)
    try:
        C.benchmark = "000001.SZ"
    except Exception:
        pass

    print(
        "loaded qmt model backtest plan",
        "signal_dates", len(signal_dates),
        "trade_dates", len(trade_dates),
        "codes", len(C.plan_codes),
        "capital", C.capital,
    )
    print("use 1m period for 145600/145700 close-sell backtest; run_time is not used")
    print("diagnostic: if no HANDLEBAR DEBUG appears, QMT did not feed 1m bars")


def handlebar(C):
    today, hhmmss = _bar_datetime(C)
    C.debug_bar_count += 1

    if C.debug_bar_count <= 10:
        try:
            timetag = C.get_bar_timetag(C.barpos)
        except Exception:
            timetag = None
        print(
            "HANDLEBAR DEBUG",
            "bar", C.debug_bar_count,
            "barpos", getattr(C, "barpos", None),
            "timetag", timetag,
            "date", today,
            "time", hhmmss,
        )

    if hhmmss == "093000" and today not in C.debug_seen_days:
        print(
            "DAY START",
            today,
            "bar", C.debug_bar_count,
            "is_signal", today in C.plan_by_signal_date,
            "is_trade", today in C.plan_by_trade_date,
        )
        C.debug_seen_days.add(today)

    if (today in C.plan_by_signal_date or today in C.plan_by_trade_date) and today not in C.debug_seen_plan_dates:
        print("HANDLEBAR PLAN DATE", today, "time", hhmmss)
        C.debug_seen_plan_dates.add(today)

    if today in C.plan_by_signal_date:
        _mark_stale_for_signal_date(C, today)

    if hhmmss >= SELL_START_TIME and today not in C.close_sell_done:
        _sell_pending(C, SELL_STYLE_COMPETE, "close_compete")
        C.close_sell_done.add(today)

    # prType=41 is lower-limit price and is sent only at/after 14:57.
    if hhmmss >= CLOSING_AUCTION_TIME and today not in C.auction_sell_done:
        _sell_pending(C, SELL_STYLE_LIMIT_DOWN, "closing_auction_limitdown")
        C.auction_sell_done.add(today)

    if today in C.plan_by_trade_date and hhmmss >= BUY_TIME:
        _buy_targets(C, today)
