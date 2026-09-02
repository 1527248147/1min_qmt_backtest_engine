#coding:utf-8
"""
ML stock-selection model backtest, v2 -- 方案B 目标仓位再平衡 (order_target).

Each rebalance every stock gets a TARGET position (equal weight 1/slots for the
top-`slots` ranked names, 0 otherwise); we trade the DELTA vs current holdings.

Execution split (per the desk's real workflow):
  * SELL side  (reductions) -- on the signal day's close: dropped names -> sell
    all, over-weight retained -> trim to target.  14:56 对手价(prType14); from
    14:57 收盘集合竞价 跌停价(prType41) -> fills at the 15:00 clearing price (NOT
    at the limit-down price).  Anything not sold stays pending and is retried
    EVERY following day until gone.  Cash sits idle until the next rebalance.
  * BUY side   (increases) -- on the trade day from 09:30: new names + under-weight
    top-ups, bought to target with current cash.  时间分散的顺延: a target that
    is limit-up / 全天停牌·退市 / 预算连最小一手都买不起 is 顺延'd to the next
    ranked candidate (replacement bought the NEXT minute, not the same minute);
    a momentary no-trade minute retries the same name until a 14:45 cutoff.

A-share board lot rules are honoured by the matcher (科创板 min 200 +1股,
北交所 min 100 +1股, 主板/创业板 100 整数倍); the strategy mirrors them for the
affordability / 顺延 decision via _round_buy.

不足一手规则: target 向下取整到板块合法手数; |delta| < 最小一手 -> 不交易 (零头
预算留现金, 不强凑); target=0 -> 清仓 (可卖零股); 新建仓预算不够最小一手 -> 顺延.

Audit logs (written on stop): suspend_log_v2.csv (每次顺延), delayed_sell_log_v2.csv
(每笔stale卖出的滞后), plus blotter_/rejects_ from run_ml_backtest.py.
"""
import csv
import datetime as dt
from pathlib import Path


PLAN_CSV = (
    "C:\\AI_STOCK\\研报\\Machine Learning for Stock Selection\\"
    "qmt_backtest\\qmt_targets_candidates.csv"
)
LOG_DIR = Path(r"C:\AI_STOCK\1min_qmt_backtest_engine\data")

ACCOUNT_ID = "testS"
FALLBACK_CAPITAL = 10000000

PRICE_STYLE = "COMPETE"
SELL_STYLE_COMPETE = 14
SELL_STYLE_LIMIT_DOWN = 41

BUY_TIME = "093000"
LIMIT_UP_CUTOFF = "130000"    # 涨停新票持续等打开, 到下午1点仍封板才顺延
SUSPEND_CUTOFF = "144500"     # after this, 顺延 still-stuck primaries to reserves
SELL_START_TIME = "145600"
CLOSING_AUCTION_TIME = "145700"

TICK = 0.01


# ---- board-aware helpers (mirror engine.account) -----------------------

def _limit_up(code, preclose):
    stk, market = code.split(".")
    market = market.upper()
    if market == "BJ":
        pct = 0.30
    elif market == "SH" and stk.startswith("688"):
        pct = 0.20
    elif market == "SZ" and stk.startswith("300"):
        pct = 0.20
    else:
        pct = 0.10
    return round(round(preclose * (1 + pct) / TICK) * TICK, 2)


def _buy_unit(code):
    stk, market = code.split(".")
    market = market.upper()
    if market == "SH" and stk.startswith("688"):
        return (200, 1)        # 科创板
    if market == "BJ":
        return (100, 1)        # 北交所
    return (100, 100)          # 主板/创业板


def _round_buy(code, shares):
    mn, step = _buy_unit(code)
    shares = int(shares)
    if shares < mn:
        return 0
    return mn + ((shares - mn) // step) * step


# ---- account / market access -------------------------------------------

def _account_total_value(C):
    try:
        for obj in get_trade_detail_data(C.account_id, "STOCK", "ACCOUNT"):
            v = getattr(obj, "m_dBalance", None)
            if v and v > 0:
                return float(v)
    except Exception:
        pass
    return float(getattr(C, "capital", 0) or getattr(C, "asset", 0) or FALLBACK_CAPITAL)


def _positions(C):
    """code -> total shares held."""
    try:
        rows = get_trade_detail_data(C.account_id, "STOCK", "POSITION")
    except Exception as e:
        print("get positions failed:", e)
        return {}
    out = {}
    for obj in rows:
        code = getattr(obj, "m_strInstrumentID", "")
        market = getattr(obj, "m_strExchangeID", "")
        vol = getattr(obj, "m_nVolume", 0)
        if code and market and vol:
            out[code + "." + market] = int(vol)
    return out


def _available(C):
    """code -> sellable shares (T+1 settled)."""
    try:
        rows = get_trade_detail_data(C.account_id, "STOCK", "POSITION")
    except Exception:
        return {}
    out = {}
    for obj in rows:
        code = getattr(obj, "m_strInstrumentID", "")
        market = getattr(obj, "m_strExchangeID", "")
        vol = getattr(obj, "m_nCanUseVolume", 0)
        if code and market and vol:
            out[code + "." + market] = int(vol)
    return out


def _quote(C, code, today, hhmmss):
    # fast path: engine get_bar_at (no pandas); QMT falls back to get_market_data_ex
    try:
        return get_bar_at(code, today, hhmmss)
    except NameError:
        pass
    try:
        data = get_market_data_ex(
            fields=["open", "high", "low", "close", "volume", "preclose"],
            stock_code=[code], period="1m",
            start_time=today + hhmmss, end_time=today + hhmmss, fill_data=False,
        )
        frame = data.get(code)
        if frame is None or len(frame) == 0:
            return None
        return frame.iloc[-1]
    except Exception:
        return None


def _trades_today(C, code, today):
    """True if the code has ANY minute bar today (停牌/退市 -> False)."""
    cache = C._data_today.setdefault(today, {})
    if code in cache:
        return cache[code]
    try:
        data = get_market_data_ex(
            fields=["close"], stock_code=[code], period="1m",
            start_time=today + "093000", end_time=today + "150000", fill_data=False,
        )
        frame = data.get(code)
        has = frame is not None and len(frame) > 0
    except Exception:
        has = False
    cache[code] = has
    return has


def _bar_datetime(C):
    timetag = C.get_bar_timetag(C.barpos)
    china = dt.datetime.utcfromtimestamp(timetag / 1000.0) + dt.timedelta(hours=8)
    return china.strftime("%Y%m%d"), china.strftime("%H%M%S")


def _load_plan(path):
    by_trade_date = {}
    by_signal_date = {}
    all_codes = set()
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            item = (int(row["rank"]), row["ts_code"], float(row["target_weight"]),
                    int(row["target_slots"]), row["signal_date"], row["trade_date"])
            by_trade_date.setdefault(row["trade_date"], []).append(item)
            by_signal_date.setdefault(row["signal_date"], []).append(item)
            all_codes.add(row["ts_code"])
    for d in by_trade_date:
        by_trade_date[d].sort()
    for d in by_signal_date:
        by_signal_date[d].sort()
    return by_signal_date, by_trade_date, all_codes


def _date_lag(C, d0, d1):
    try:
        return len([d for d in C._trade_days if d0 < d <= d1])
    except Exception:
        return 0


def _log_suspend(C, trade_date, hhmmss, rank, code, reason):
    C.suspend_log.append({"trade_date": trade_date, "time": hhmmss,
                          "rank": rank, "code": code, "reason": reason})


def _log_cleared(C, code, today, hhmmss):
    mark = C.sell_mark.pop(code, (None, today, hhmmss))
    C.delayed_sell_log.append({
        "code": code, "signal_date": mark[0],
        "mark_date": mark[1], "mark_time": mark[2],
        "sell_date": today, "sell_time": hhmmss,
        "lag_days": _date_lag(C, mark[1], today),
    })


# ---- SELL side: target reductions, per-bar + cross-day retry ------------

def _plan_sells(C, signal_date, today, hhmmss):
    """At the signal day's close, set pending_sells = {code: target_shares} for
    every held name that must be reduced (dropped -> 0, over-weight -> trim)."""
    if signal_date in C.stale_marked_dates:
        return
    rows = C.plan_by_signal_date.get(signal_date)
    if not rows:
        return
    slots = rows[0][3]
    target_set = set(code for (rank, code, w, sl, sd, td) in rows if rank <= slots)
    w = 1.0 / slots
    nav = _account_total_value(C)
    held = _positions(C)

    new_pending = {}
    for code, cur in held.items():
        if code in target_set:
            q = _quote(C, code, today, hhmmss)
            if q is None or q["close"] <= 0:
                continue                       # 停牌, can't trim now -> keep
            tgt = min(_round_buy(code, nav * w / q["close"]), cur)
            if cur > tgt:
                new_pending[code] = tgt        # trim over-weight down to target
        else:
            new_pending[code] = 0              # dropped -> sell all

    for code in new_pending:
        if code not in C.sell_mark:
            C.sell_mark[code] = (signal_date, today, hhmmss)
    for code in list(C.sell_mark):
        if code not in new_pending:            # re-retained or already at target
            C.sell_mark.pop(code, None)
    C.pending_sells = new_pending
    C.stale_marked_dates.add(signal_date)
    print("plan sells", signal_date, "reduce", len(new_pending))


def _reap_sells(C, today, hhmmss):
    held = _positions(C)
    for code in list(C.pending_sells):
        if held.get(code, 0) <= C.pending_sells[code]:
            _log_cleared(C, code, today, hhmmss)
            C.pending_sells.pop(code)


def _run_sells(C, prtype, tag, today, hhmmss):
    _reap_sells(C, today, hhmmss)              # log fills from prior bars/days
    if not C.pending_sells:
        return
    held = _positions(C)
    avail = _available(C)
    sent = 0
    for code, tgt in sorted(C.pending_sells.items()):
        cur = held.get(code, 0)
        can = avail.get(code, 0)
        if cur <= tgt or can <= 0:
            continue
        qty = min(cur - tgt, can)
        remark = tag + "_" + code.replace(".", "_")
        try:
            passorder(24, 1101, C.account_id, code, prtype, -1, qty, tag, 2, remark, C)
        except TypeError:
            passorder(24, 1101, C.account_id, code, prtype, -1, qty, C)
        sent += 1
    _reap_sells(C, today, hhmmss)              # log same-bar fills (synchronous)
    if sent:
        print("sell", tag, "names", sent, "prtype", prtype)


# ---- BUY side: target increases with time-spread 顺延 ------------------

def _run_buys(C, trade_date, hhmmss):
    rows = C.plan_by_trade_date.get(trade_date)
    if not rows or trade_date in C.buy_done_dates:
        return
    slots = rows[0][3]
    w = 1.0 / slots

    st = C.buy_state.get(trade_date)
    if st is None:
        st = {"queue_i": 0, "active": [], "filled": set(),
              "rank_of": {item[1]: item[0] for item in rows}}
        C.buy_state[trade_date] = st

    held = _positions(C)
    nav = _account_total_value(C)
    at_cutoff = hhmmss >= SUSPEND_CUTOFF

    def pull_next():
        i = st["queue_i"]
        while i < len(rows):
            code = rows[i][1]
            i += 1
            if code in st["filled"] or code in st["active"]:
                continue
            st["queue_i"] = i
            return code
        st["queue_i"] = i
        return None

    while len(st["filled"]) + len(st["active"]) < slots:
        nxt = pull_next()
        if nxt is None:
            break
        st["active"].append(nxt)

    for code in list(st["active"]):
        cur = held.get(code, 0)
        rank = st["rank_of"].get(code, -1)
        unit = _buy_unit(code)[0]
        q = _quote(C, code, trade_date, hhmmss)
        tgt = _round_buy(code, nav * w / q["close"]) if (q is not None and q["close"] > 0) else 0

        # reached target (within <1 lot): slot done, no more buying
        if cur > 0 and (tgt - cur) < unit:
            st["filled"].add(code); st["active"].remove(code); continue
        # new name, budget < 1 lot -> unaffordable -> 顺延
        if cur <= 0 and tgt <= 0:
            st["active"].remove(code); _log_suspend(C, trade_date, hhmmss, rank, code, "unaffordable")
            continue

        # 涨停封板: 新建仓持续每分钟重试等打开, 到下午1点仍封板才顺延;
        # 已部分持有的留着(收盘前接受部分)
        if q is not None and q["volume"] > 0 and q["low"] >= _limit_up(code, q["preclose"]) - TICK / 2:
            if cur <= 0:
                if hhmmss >= LIMIT_UP_CUTOFF:
                    st["active"].remove(code); _log_suspend(C, trade_date, hhmmss, rank, code, "limit_up_13h")
                # else 保持active, 等涨停打开
            elif at_cutoff:
                st["filled"].add(code); st["active"].remove(code)
            continue
        # 无bar/无量: 退市新票立即顺延; 在交易的下一分钟重试; cutoff兜底
        if q is None or q["volume"] == 0:
            if cur <= 0 and not _trades_today(C, code, trade_date):
                st["active"].remove(code); _log_suspend(C, trade_date, hhmmss, rank, code, "no_data_today")
            elif at_cutoff:
                st["active"].remove(code)
                if cur > 0:
                    st["filled"].add(code)
                else:
                    _log_suspend(C, trade_date, hhmmss, rank, code, "no_trade_at_cutoff")
            continue

        # 可交易: 朝目标买入, 单分钟受 max_vol_rate 限量则部分成交, 下一分钟继续累积
        got = order_shares(code, tgt - cur, PRICE_STYLE, -1, C, C.account_id)
        new_cur = cur + (got or 0)
        if (tgt - new_cur) < unit:
            st["filled"].add(code); st["active"].remove(code)        # reached target
        elif at_cutoff:
            st["active"].remove(code)
            if new_cur > 0:
                st["filled"].add(code)
            else:
                _log_suspend(C, trade_date, hhmmss, rank, code, "order_fail")
        # else: keep active, accumulate over the next minute(s)

    while len(st["filled"]) + len(st["active"]) < slots:
        nxt = pull_next()
        if nxt is None:
            break
        st["active"].append(nxt)

    if len(st["filled"]) >= slots or (not st["active"] and st["queue_i"] >= len(rows)):
        C.buy_done_dates.add(trade_date)
        print("buy done", trade_date, hhmmss, "filled", len(st["filled"]), "slots", slots)


# ---- lifecycle ----------------------------------------------------------

def init(C):
    C.account_id = ACCOUNT_ID
    C.plan_by_signal_date, C.plan_by_trade_date, C.plan_codes = _load_plan(PLAN_CSV)
    C.pending_sells = {}            # code -> target_shares (sell down to this)
    C.stale_marked_dates = set()
    C.buy_done_dates = set()
    C.buy_state = {}
    C.sell_mark = {}
    C._data_today = {}
    C.suspend_log = []
    C.delayed_sell_log = []
    C._trade_days = sorted(C.plan_by_trade_date)

    sig = sorted(C.plan_by_signal_date)
    trd = sorted(C.plan_by_trade_date)
    if trd:
        C.start_time = sig[0]
        C.end_time = trd[-1]
    try:
        C.set_universe(["000001.SZ"])
        C.benchmark = "000001.SZ"
    except Exception:
        pass
    print("v2 (方案B) plan loaded: signal_dates", len(sig), "trade_dates", len(trd),
          "candidate_codes", len(C.plan_codes))


def handlebar(C):
    today, hhmmss = _bar_datetime(C)

    # decide reductions at the signal day's close, then sell every bar/day
    if today in C.plan_by_signal_date and hhmmss >= SELL_START_TIME:
        _plan_sells(C, today, today, hhmmss)
    if SELL_START_TIME <= hhmmss < CLOSING_AUCTION_TIME:
        _run_sells(C, SELL_STYLE_COMPETE, "close_compete", today, hhmmss)
    elif hhmmss >= CLOSING_AUCTION_TIME:
        _run_sells(C, SELL_STYLE_LIMIT_DOWN, "closing_auction_limitdown", today, hhmmss)

    if today in C.plan_by_trade_date and hhmmss >= BUY_TIME:
        _run_buys(C, today, hhmmss)


def stop(C):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "suspend_log_v2.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trade_date", "time", "rank", "code", "reason"])
        w.writeheader(); w.writerows(C.suspend_log)
    fields = ["code", "signal_date", "mark_date", "mark_time", "sell_date", "sell_time", "lag_days"]
    with (LOG_DIR / "delayed_sell_log_v2.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(C.delayed_sell_log)
        for c, m in C.sell_mark.items():       # never sold by end of backtest
            w.writerow({"code": c, "signal_date": m[0], "mark_date": m[1],
                        "mark_time": m[2], "sell_date": "", "sell_time": "",
                        "lag_days": "UNSOLD"})
    print(f"audit: suspend {len(C.suspend_log)}, delayed_sell {len(C.delayed_sell_log)}, "
          f"unsold {len(C.sell_mark)}")
