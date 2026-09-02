#coding:utf-8
"""
ML stock-selection model backtest -- TWAP execution variant.

Same 方案B target-rebalance + 顺延 + corporate-action-correct accounting as
qmt_ml_signal_model_backtest_v2.py, but buys and sells are sliced evenly over a
time window (TWAP) instead of "buy ASAP at open / sell at close".  The v2 file is
left untouched -- run whichever you want via --strategy.

TWAP口径 (mirrors QMT 算法交易 parameters; the real splitting logic is closed C++
on the broker side, so it is reimplemented here):
  * 时间窗: buy over [BUY_START, BUY_END], sell over [SELL_START, SELL_END] each day
  * 量比比例 PARTICIPATION: 单分钟最多吃该股当根bar成交量的这个比例 (QMT 默认 20%)
  * 委托最小金额 MIN_ORDER_AMT: 小于此金额的切片跳过 (临近窗口末端强制完成)
  * 对手价成交, 涨停不买/跌停不卖, 退市/停牌顺延 (与 v2 一致)

Cumulative TWAP target at time t over a window of S session-minutes:
  buy:  hold_target(t)   = final_target * (elapsed / S)         (build up)
  sell: hold_target(t)   = cur0 - (cur0 - final) * (elapsed / S) (wind down)
each bar trades the delta toward hold_target(t), capped by PARTICIPATION*volume.
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

# ---- TWAP windows / params ----
BUY_START, BUY_END = "093000", "140000"      # 次日买入TWAP窗口
SELL_START, SELL_END = "093000", "145700"    # signal日卖出TWAP窗口
PARTICIPATION = 0.20                          # 量比上限: 单bar最多吃该股成交量的20%
MIN_ORDER_AMT = 10000.0                       # 委托最小金额(元), 太小的切片跳过
LIMIT_UP_CUTOFF = "130000"                    # 涨停新票等到13:00仍封板才顺延

TICK = 0.01


# ---- board-aware helpers ------------------------------------------------

def _limit_up(code, preclose):
    stk, market = code.split("."); market = market.upper()
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
    stk, market = code.split("."); market = market.upper()
    if market == "SH" and stk.startswith("688"):
        return (200, 1)
    if market == "BJ":
        return (100, 1)
    return (100, 100)


def _round_buy(code, shares):
    mn, step = _buy_unit(code)
    shares = int(shares)
    if shares < mn:
        return 0
    return mn + ((shares - mn) // step) * step


def _sess_min(hhmmss):
    """Session-minute index from 09:30 (0) to 15:00 (240), skipping the lunch gap."""
    t = int(hhmmss[:2]) * 60 + int(hhmmss[2:4])
    if t <= 690:                      # 09:30..11:30
        return min(120, max(0, t - 570))
    return min(240, 120 + (t - 780))  # 13:01..15:00


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
    try:
        rows = get_trade_detail_data(C.account_id, "STOCK", "POSITION")
    except Exception:
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
            start_time=today + hhmmss, end_time=today + hhmmss, fill_data=False)
        frame = data.get(code)
        if frame is None or len(frame) == 0:
            return None
        return frame.iloc[-1]
    except Exception:
        return None


def _trades_today(C, code, today):
    cache = C._data_today.setdefault(today, {})
    if code in cache:
        return cache[code]
    try:
        data = get_market_data_ex(fields=["close"], stock_code=[code], period="1m",
                                  start_time=today + "093000", end_time=today + "150000",
                                  fill_data=False)
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
    by_trade_date, by_signal_date, all_codes = {}, {}, set()
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
    C.delayed_sell_log.append({"code": code, "signal_date": mark[0],
                               "mark_date": mark[1], "mark_time": mark[2],
                               "sell_date": today, "sell_time": hhmmss,
                               "lag_days": _date_lag(C, mark[1], today)})


# ---- SELL side: TWAP wind-down over the day, retried across days ---------

def _plan_sells(C, signal_date, today, hhmmss):
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
                continue
            tgt = min(_round_buy(code, nav * w / q["close"]), cur)
            if cur > tgt:
                new_pending[code] = tgt
        else:
            new_pending[code] = 0
    for code in new_pending:
        if code not in C.sell_mark:
            C.sell_mark[code] = (signal_date, today, hhmmss)
    for code in list(C.sell_mark):
        if code not in new_pending:
            C.sell_mark.pop(code, None)
    C.pending_sells = new_pending
    C.sell_cur0 = dict(held)          # holding at the start of the sell day (TWAP base)
    C.stale_marked_dates.add(signal_date)
    print("plan sells", signal_date, "reduce", len(new_pending))


def _run_sells(C, today, hhmmss):
    if not C.pending_sells:
        return
    s = _sess_min(hhmmss); S = _sess_min(SELL_END)
    frac = 1.0 if s >= S else max(0.0, s / S)
    held = _positions(C)
    avail = _available(C)
    # reap already-completed
    for code in list(C.pending_sells):
        if held.get(code, 0) <= C.pending_sells[code]:
            _log_cleared(C, code, today, hhmmss); C.pending_sells.pop(code, None)
    sent = 0
    for code, tgt in sorted(C.pending_sells.items()):
        cur = held.get(code, 0)
        can = avail.get(code, 0)
        if cur <= tgt or can <= 0:
            continue
        cur0 = C.sell_cur0.get(code, cur)
        hold_target = cur0 - (cur0 - tgt) * frac        # how much we may still hold now
        want_sell = int(cur - max(tgt, hold_target))    # cumulative TWAP sell so far
        if want_sell < 100 and frac < 1.0:
            continue
        q = _quote(C, code, today, hhmmss)
        cap = int(PARTICIPATION * q["volume"]) if q is not None else 0
        qty = min(want_sell, can, cap if frac < 1.0 else can)
        if qty <= 0:
            continue
        remark = "twap_sell_" + code.replace(".", "_")
        try:
            passorder(24, 1101, C.account_id, code, SELL_STYLE_COMPETE, -1, qty, "twap_sell", 2, remark, C)
        except TypeError:
            passorder(24, 1101, C.account_id, code, SELL_STYLE_COMPETE, -1, qty, C)
        sent += 1
    held2 = _positions(C)
    for code in list(C.pending_sells):
        if held2.get(code, 0) <= C.pending_sells[code]:
            _log_cleared(C, code, today, hhmmss); C.pending_sells.pop(code, None)
    if sent:
        print("twap sell", today, hhmmss, "names", sent)


# ---- BUY side: TWAP build-up over the day, with 顺延 --------------------

def _run_buys(C, trade_date, hhmmss):
    rows = C.plan_by_trade_date.get(trade_date)
    if not rows or trade_date in C.buy_done_dates:
        return
    slots = rows[0][3]; w = 1.0 / slots
    st = C.buy_state.get(trade_date)
    if st is None:
        st = {"queue_i": 0, "active": [], "filled": set(),
              "rank_of": {item[1]: item[0] for item in rows}}
        C.buy_state[trade_date] = st

    held = _positions(C)
    nav = _account_total_value(C)
    s = _sess_min(hhmmss); S = _sess_min(BUY_END)
    frac = 1.0 if s >= S else max(0.0, s / S)
    eod = hhmmss >= "150000"     # 收盘最后一根bar才接受未买满/放弃; 此前一直补

    def pull_next():
        i = st["queue_i"]
        while i < len(rows):
            code = rows[i][1]; i += 1
            if code in st["filled"] or code in st["active"]:
                continue
            st["queue_i"] = i; return code
        st["queue_i"] = i; return None

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

        # reached final target -> slot done
        if cur > 0 and (tgt - cur) < unit:
            st["filled"].add(code); st["active"].remove(code); continue
        if cur <= 0 and tgt <= 0:
            st["active"].remove(code); _log_suspend(C, trade_date, hhmmss, rank, code, "unaffordable"); continue
        # 涨停: 新票等到13:00仍封板才顺延; 已持有的留着重试, 收盘才接受部分
        if q is not None and q["volume"] > 0 and q["low"] >= _limit_up(code, q["preclose"]) - TICK / 2:
            if cur <= 0 and hhmmss >= LIMIT_UP_CUTOFF:
                st["active"].remove(code); _log_suspend(C, trade_date, hhmmss, rank, code, "limit_up_13h")
            elif cur > 0 and eod:
                st["filled"].add(code); st["active"].remove(code)
            continue
        # 无bar/无量: 退市新票顺延; 在交易则下一分钟重试; 收盘兜底
        if q is None or q["volume"] == 0:
            if cur <= 0 and not _trades_today(C, code, trade_date):
                st["active"].remove(code); _log_suspend(C, trade_date, hhmmss, rank, code, "no_data_today")
            elif eod:
                st["active"].remove(code)
                if cur > 0:
                    st["filled"].add(code)
                else:
                    _log_suspend(C, trade_date, hhmmss, rank, code, "no_trade")
            continue

        # TWAP cumulative target by now (after 14:00 = full target); 量比上限始终生效.
        # 某单部分成交/受限 -> 留 active, 下一分钟继续补, 直到买满或收盘.
        twap_target = _round_buy(code, tgt * frac)
        delta = twap_target - cur
        if delta < unit or (delta * q["close"] < MIN_ORDER_AMT and not eod):
            if eod and cur > 0:
                st["filled"].add(code); st["active"].remove(code)
            continue
        cap = int(PARTICIPATION * q["volume"])
        buy_qty = _round_buy(code, min(delta, cap))
        if buy_qty < unit:
            if eod and cur > 0:
                st["filled"].add(code); st["active"].remove(code)
            continue
        got = order_shares(code, buy_qty, PRICE_STYLE, -1, C, C.account_id)
        new_cur = cur + (got or 0)
        if (tgt - new_cur) < unit:
            st["filled"].add(code); st["active"].remove(code)
        elif eod:
            st["active"].remove(code)
            if new_cur > 0:
                st["filled"].add(code)
            else:
                _log_suspend(C, trade_date, hhmmss, rank, code, "order_fail")

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
    C.pending_sells = {}
    C.sell_cur0 = {}
    C.stale_marked_dates = set()
    C.buy_done_dates = set()
    C.buy_state = {}
    C.sell_mark = {}
    C._data_today = {}
    C.suspend_log = []
    C.delayed_sell_log = []
    C._trade_days = sorted(C.plan_by_trade_date)
    sig = sorted(C.plan_by_signal_date); trd = sorted(C.plan_by_trade_date)
    if trd:
        C.start_time = sig[0]; C.end_time = trd[-1]
    try:
        C.set_universe(["000001.SZ"]); C.benchmark = "000001.SZ"
    except Exception:
        pass
    print("TWAP plan loaded: signal_dates", len(sig), "trade_dates", len(trd),
          "codes", len(C.plan_codes), "| buy", BUY_START, "-", BUY_END,
          "sell", SELL_START, "-", SELL_END, "量比", PARTICIPATION)


def handlebar(C):
    today, hhmmss = _bar_datetime(C)
    if today in C.plan_by_signal_date and hhmmss >= SELL_START:
        _plan_sells(C, today, today, hhmmss)
    if SELL_START <= hhmmss <= SELL_END:
        _run_sells(C, today, hhmmss)
    if today in C.plan_by_trade_date and BUY_START <= hhmmss:
        _run_buys(C, today, hhmmss)


def stop(C):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "suspend_log_twap.csv").open("w", encoding="utf-8-sig", newline="") as f:
        wsv = csv.DictWriter(f, fieldnames=["trade_date", "time", "rank", "code", "reason"])
        wsv.writeheader(); wsv.writerows(C.suspend_log)
    fields = ["code", "signal_date", "mark_date", "mark_time", "sell_date", "sell_time", "lag_days"]
    with (LOG_DIR / "delayed_sell_log_twap.csv").open("w", encoding="utf-8-sig", newline="") as f:
        wsv = csv.DictWriter(f, fieldnames=fields); wsv.writeheader(); wsv.writerows(C.delayed_sell_log)
        for c, m in C.sell_mark.items():
            wsv.writerow({"code": c, "signal_date": m[0], "mark_date": m[1], "mark_time": m[2],
                          "sell_date": "", "sell_time": "", "lag_days": "UNSOLD"})
    print(f"audit: suspend {len(C.suspend_log)}, delayed_sell {len(C.delayed_sell_log)}, unsold {len(C.sell_mark)}")
