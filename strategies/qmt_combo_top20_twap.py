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


# Plan file. Overridable so the same (already corrected) strategy can be run
# against the top-2.5% / 5% / 10% baskets without cloning the file and having
# the full-liquidation fix drift out of sync between copies.
import os as _os
PLAN_CSV = _os.environ.get("COMBO_PLAN_CSV") or (
    "C:\\AI_STOCK\\machine_learning_stock_selection\\"
    "3windows_18models_alpha192+fund310+forecast\\automatically_plan_generate\\"
    "qmt_plan_recent_seasonal_long10_top20.csv"
)
LOG_DIR = Path(r"C:\AI_STOCK\1min_qmt_backtest_engine\data")

ACCOUNT_ID = "testS"
FALLBACK_CAPITAL = 10000000

PRICE_STYLE = "COMPETE"
SELL_STYLE_COMPETE = 14

# ---- TWAP windows / params ----
BUY_START, BUY_END = "093000", "140000"      # 次日买入TWAP窗口
SELL_START, SELL_END = "093000", "140000"    # signal日卖出TWAP窗口
# 14:00 之后进入加速清仓段，目标拉满、取消最小切片，量比上限从 10% 放宽到 30%。
# 实盘脚本（combo_sell_dual_model.py）还会在 14:56 撤光挂单、14:57 把剩余量挂跌停价进收盘集合竞价；
# 集合竞价无法在这里建模（1 分钟 K 线不含竞价数据），所以回测会低估最终完成度。
SELL_RUSH_END = "145700"                     # 加速段终点（即原来的卖出窗口末端）
PARTICIPATION_RUSH = 0.30                    # 加速段量比上限
PARTICIPATION = 0.10                          # 量比上限: 单bar最多吃该股成交量的10%
MIN_ORDER_AMT = 2000.0                        # 委托最小金额(元): 20万/20只≈1万/票, 调低使TWAP真正拆单
LIMIT_UP_CUTOFF = "130000"                    # 涨停新票等到13:00仍封板才顺延
NO_TRADE_CUTOFF = "100000"                    # 无量新票: 10:00前一直重试; 10:00后回看09:30~当下仍零成交才顺延(点位时刻,无未来函数)

TICK = 0.01


# ---- board-aware helpers ------------------------------------------------

_ST_RANGES = None


def _st_ranges():
    """code -> ((start, end), ...) from data/st_ranges.csv, same file the engine
    reads. Duplicated rather than imported because this file is also pasted into
    QMT, where `engine` does not exist. Missing file -> {} -> board limits."""
    global _ST_RANGES
    if _ST_RANGES is None:
        _ST_RANGES = {}
        try:
            with (LOG_DIR / "st_ranges.csv").open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    _ST_RANGES.setdefault(row["ts_code"], []).append(
                        (int(row["start_date"]), int(row["end_date"])))
        except Exception:
            _ST_RANGES = {}
    return _ST_RANGES


def _is_st(code, trade_date):
    rng = _ST_RANGES.get(code) if _ST_RANGES is not None else _st_ranges().get(code)
    if not rng:
        return False
    d = int(trade_date)
    return any(a <= d <= b for a, b in rng)


def _limit_up(code, preclose, trade_date=None):
    """Ceiling price. MUST agree with engine/account.py limit_prices(), or the
    strategy's "give up at 13:00 / accept the partial" decisions get made on a
    different board than the fills do.

    Two bugs fixed here on 2026-08-06:
      * 301xxx.SZ was missing. It is ChiNext, +-20%, but only `300` was tested,
        so every 301 name was priced at the main-board +-10% -- a ceiling 10%
        too low, which reads as "limit-up, cannot buy" on any strong day. This
        month's basket holds 301503.SZ.
      * ST/*ST is +-5% and was not considered at all.
    """
    _st_ranges()
    if trade_date is not None and _is_st(code, trade_date):
        pct = 0.05
    else:
        stk, market = code.split("."); market = market.upper()
        if market == "BJ":
            pct = 0.30
        elif market == "SH" and stk.startswith("688"):
            pct = 0.20
        elif market == "SZ" and (stk.startswith("300") or stk.startswith("301")):
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


def _round_sell(code, shares, clearing):
    """把卖出量取整成合法委托量，在 passorder 之前。

    规则同 engine/account.py 的 round_sell()：
      主板/创业板  拆单必须是 100 的整数倍，且不低于 100
      科创板       步长 1 股，但拆单不得低于 200 股
      北交所       步长 1 股，但拆单不得低于 100 股
      清仓（卖出量 >= 可用量）时允许带零头

    科创板/北交所那两个下限以前漏了，回测因此允许 75 股的科创板拆单成交，
    而实盘发不出去（规定余额不足 200 股必须一次性卖出）。以实盘为准。

    以前这里不取整，直接把 min(want_sell, can, cap) 发出去。引擎收到后照样会
    取整，所以并不会多卖 —— 但一个 75 股的委托会被取整成 0，然后记一笔
    volume_cap 拒单。回测里只是个计数，实盘里那是一笔真实发出的废单：0.1 元
    流量费，外加交易所看到的委托笔数。不合规的量本来就不该发出去。
    """
    shares = int(shares)
    if clearing:
        return shares
    mn, step = _buy_unit(code)
    shares = (shares // step) * step
    return shares if shares >= mn else 0


def _round_buy(code, shares):
    mn, step = _buy_unit(code)
    shares = int(shares)
    if shares < mn:
        return 0
    return mn + ((shares - mn) // step) * step


def _prev_min(hhmmss):
    """Label of the previous minute, hopping the 11:30-13:00 lunch gap."""
    t = int(hhmmss[:2]) * 60 + int(hhmmss[2:4]) - 1
    if 11 * 60 + 30 < t < 13 * 60:
        t = 11 * 60 + 30
    return "%02d%02d00" % (t // 60, t % 60)


def _cap_volume(C, code, today, hhmmss):
    """Volume the participation cap may be computed from: the PREVIOUS completed
    minute, never the one being traded.

    The cap used to read q["volume"] of the CURRENT bar -- the full minute's
    volume, while still trading inside that minute. That is not knowable in real
    time, and it flatters exactly the moments that matter: the minute a sealed
    board cracks, the backtest sees the whole burst and sizes against it, while
    the live script only learns of it a minute later. The live scripts have
    always used _prev_min for this; the backtest did not, and the gap showed up
    as the backtest being able to accumulate limit-up names that live cannot.

    None (no previous bar, e.g. the first minute of the session) -> 0, and the
    one-lot floor below decides what still gets through.
    """
    pq = _quote(C, code, today, _prev_min(hhmmss))
    if pq is None:
        return 0
    try:
        return int(pq["volume"])
    except Exception:
        return 0


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


def _traded_since_open(C, code, today, hhmmss):
    """点位时刻: code 在 09:30~当前这根bar之间有没有成交过? 只回看过去, 无未来函数.
    单调: 一旦见到成交, 当天永久缓存 True, 后续不再查询."""
    seen = C._data_today.setdefault(today, set())
    if code in seen:
        return True
    try:
        data = get_market_data_ex(fields=["close"], stock_code=[code], period="1m",
                                  start_time=today + "093000", end_time=today + hhmmss,
                                  fill_data=False)
        frame = data.get(code)
        has = frame is not None and len(frame) > 0
    except Exception:
        has = False
    if has:
        seen.add(code)
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
    # FULL LIQUIDATION on the signal day -- every held name goes to zero.
    #
    # The previous version kept whatever was also in the incoming basket:
    #     if code in target_set: tgt = min(target_shares, cur)   # hold it
    #     else:                  tgt = 0                          # sell it
    # That is what live trading cannot do. The incoming basket comes from a
    # signal computed at THIS day's close (config.py: "signal = monthly trade-day
    # 0 close | entry = 1d open"), so at 09:30, when the sell TWAP starts, the
    # new names are not knowable yet. Measured on the 2026 blotter, it kept 1-3
    # names per rebalance -- 20260401 sold 17 of 20, 20260506 and 20260601 sold
    # 18 of 20 -- while the live scripts sell all 20 and re-buy the overlap the
    # next day. Roughly 9% of NAV of extra round-trip that the backtest never
    # paid for, worth about 0.13pp a year once slippage is priced in.
    #
    # This is not a bug introduced with the top-20 variant: the identical block
    # exists in qmt_ml_signal_model_backtest_twap.py from 2026-06-20. The whole
    # published 599% / 27% ann series carries it.
    held = _positions(C)
    new_pending = dict((code, 0) for code in held)
    for code in new_pending:
        if code not in C.sell_mark:
            C.sell_mark[code] = (signal_date, today, hhmmss)
    for code in list(C.sell_mark):
        if code not in new_pending:
            C.sell_mark.pop(code, None)
    C.pending_sells = new_pending
    C.sell_cur0 = dict(held)          # holding at the start of the sell day (TWAP base)
    C.sell_cur0_date = today          # so _run_sells does not re-anchor again today
    C.stale_marked_dates.add(signal_date)
    print("plan sells", signal_date, "reduce", len(new_pending))


def _run_sells(C, today, hhmmss):
    if not C.pending_sells:
        return
    s = _sess_min(hhmmss); S = _sess_min(SELL_END)
    frac = 1.0 if s >= S else max(0.0, s / S)
    held = _positions(C)
    avail = _available(C)
    # RE-ANCHOR EACH DAY. sell_cur0 is the denominator of the TWAP curve, and it
    # used to be set once, on the signal day, and never again. A name that could
    # not be fully sold that day then carried a curve anchored on its ORIGINAL
    # holding into every following day: at 09:30 hold_target == cur0, which is
    # above what is actually left, so want_sell came out negative and nothing
    # was sold. Selling only resumed once frac had climbed past the fraction
    # already sold -- a name 60% done sat idle until roughly 13:22, and a name
    # 90% done until 14:40. The engine was replaying day one's curve instead of
    # trading day two.
    #
    # The live script (combo_sell_close_model.py) does not do this: each session
    # computes allowed and frac fresh, so whatever is still held is spread over
    # that whole day, 09:30 to 14:57. Re-anchoring here makes the backtest match
    # the thing actually being run, and it is also the better rule -- when a
    # limit-down opens or a halt lifts, you want to be selling from 09:30, not
    # waiting for a curve fitted to a day that has already passed.
    if C.sell_cur0_date != today:
        C.sell_cur0 = dict(held)
        C.sell_cur0_date = today
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
        # Previous completed minute, same reason as the buy side. Fixing only
        # one side would make selling look easier than buying and bias the
        # result; this is a correctness fix, not a policy change.
        # 量比上限不再“到点取消”：原来 frac>=1 就完全不限，等于在
        # 清淡盘口上无限量砸。现在剩余量由收盘竞价承接（实盘），
        # 加速段只把 10% 放宽到 30%。
        _part = PARTICIPATION_RUSH if hhmmss >= SELL_END else PARTICIPATION
        cap = int(_part * _cap_volume(C, code, today, hhmmss))
        qty = min(want_sell, can, cap)
        # clearing 的判定和引擎一致：是在按 can_use 和量比封顶【之后】才比较的，
        # 所以这里也必须用封顶后的 qty 去比，否则两边对“算不算清仓”的看法会不同。
        qty = _round_sell(code, qty, qty >= can)
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

        # 无bar/无量必须先判: 缺行情 != 买不起(否则 tgt=0 会被误判成 unaffordable).
        # 10:00前一直重试; 10:00后回看09:30~当下仍零成交(整段停牌)才顺延; 收盘兜底
        # VOLUME from the previous completed minute, price from this one.
        # "This minute has no volume" is not knowable while trading inside it --
        # the same lookahead the cap had. Live (combo_buy_open_model.py) reads
        # _prev_min for exactly this test; the backtest read the current bar and
        # therefore silently skipped minutes it could not have known were dead.
        if q is None or q["close"] <= 0 or _cap_volume(C, code, trade_date, hhmmss) == 0:
            if (cur <= 0 and hhmmss >= NO_TRADE_CUTOFF
                    and not _traded_since_open(C, code, trade_date, hhmmss)):
                st["active"].remove(code); _log_suspend(C, trade_date, hhmmss, rank, code, "no_trade_by_1000")
            elif eod:
                st["active"].remove(code)
                if cur > 0:
                    st["filled"].add(code)
                else:
                    _log_suspend(C, trade_date, hhmmss, rank, code, "no_trade")
            continue

        # 行情有效, 目标量才有意义
        tgt = _round_buy(code, nav * w / q["close"])

        # reached final target -> slot done
        if cur > 0 and (tgt - cur) < unit:
            st["filled"].add(code); st["active"].remove(code); continue
        if cur <= 0 and tgt <= 0:          # 确实一手都买不起
            st["active"].remove(code); _log_suspend(C, trade_date, hhmmss, rank, code, "unaffordable"); continue
        # 涨停: 新票等到13:00仍封板才顺延; 已持有的留着重试, 收盘才接受部分
        # CLOSE, not LOW -- must match engine/account.py can_buy. A minute that
        # CLOSES on the ceiling is a minute we would have been buying the seal.
        if (q["volume"] > 0
                and q["close"] >= _limit_up(code, q["preclose"], trade_date) - TICK / 2):
            if cur <= 0 and hhmmss >= LIMIT_UP_CUTOFF:
                st["active"].remove(code); _log_suspend(C, trade_date, hhmmss, rank, code, "limit_up_13h")
            elif cur > 0 and eod:
                st["filled"].add(code); st["active"].remove(code)
            continue

        # TWAP cumulative target by now (after 14:00 = full target); 量比上限始终生效.
        # 某单部分成交/受限 -> 留 active, 下一分钟继续补, 直到买满或收盘.
        twap_target = _round_buy(code, tgt * frac)
        delta = twap_target - cur
        if delta < unit or (delta * q["close"] < MIN_ORDER_AMT and not eod):
            if eod and cur > 0:
                st["filled"].add(code); st["active"].remove(code)
            continue
        # (2) Cap DROPPED once the TWAP window is over, matching the sell side
        #     (`cap if frac < 1.0 else can`). Keeping it past BUY_END means a
        #     thin name can never be completed, however long it is given.
        # (3) FLOOR of one tradable lot. cap used to fall to e.g. 10 shares on a
        #     1-lot minute; _round_buy then returns 0 and the name does nothing,
        #     all day. One lot a minute is still bounded (240 lots a session).
        if frac >= 1.0:
            cap = delta
        else:
            cap = int(PARTICIPATION * _cap_volume(C, code, trade_date, hhmmss))
            cap = max(cap, unit)
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
    C.sell_cur0_date = None    # the day sell_cur0 describes; re-anchored daily
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
    if SELL_START <= hhmmss <= SELL_RUSH_END:
        _run_sells(C, today, hhmmss)
    if today in C.plan_by_trade_date and BUY_START <= hhmmss:
        _run_buys(C, today, hhmmss)


def stop(C):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "suspend_log_combo_top20.csv").open("w", encoding="utf-8-sig", newline="") as f:
        wsv = csv.DictWriter(f, fieldnames=["trade_date", "time", "rank", "code", "reason"])
        wsv.writeheader(); wsv.writerows(C.suspend_log)
    fields = ["code", "signal_date", "mark_date", "mark_time", "sell_date", "sell_time", "lag_days"]
    with (LOG_DIR / "delayed_sell_log_combo_top20.csv").open("w", encoding="utf-8-sig", newline="") as f:
        wsv = csv.DictWriter(f, fieldnames=fields); wsv.writeheader(); wsv.writerows(C.delayed_sell_log)
        for c, m in C.sell_mark.items():
            wsv.writerow({"code": c, "signal_date": m[0], "mark_date": m[1], "mark_time": m[2],
                          "sell_date": "", "sell_time": "", "lag_days": "UNSOLD"})
    print(f"audit: suspend {len(C.suspend_log)}, delayed_sell {len(C.delayed_sell_log)}, unsold {len(C.sell_mark)}")
