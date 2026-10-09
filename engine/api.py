#coding:utf-8
"""
Global trade/data functions injected into the strategy's namespace.

These reproduce the QMT GUI model-backtest globals the ML strategy calls:

    get_trade_detail_data(account, datatype, datakind)
    passorder(opType, orderType, accountid, code, prType, modelprice, volume, ... , C)
    order_target_value(code, value, priceType, C, accountid)
    order_target_percent(code, percent, priceType, C, accountid)
    order_shares(code, shares, priceType, price, C, accountid)
    get_market_data_ex(...)
    timetag_to_datetime(timetag, format)

They are closures bound to a single Engine instance so they share its account,
data feed and "current bar" cursor.  ``build_globals(engine)`` returns the dict
the runner injects.
"""

from __future__ import annotations

import datetime as dt

from .datafeed import beijing_to_timetag

# QMT opType: 23 = stock buy, 24 = stock sell
OP_BUY = 23
OP_SELL = 24

_BAR_FIELDS = ("open", "high", "low", "close", "volume", "preclose")


def timetag_to_datetime(timetag, fmt="%Y%m%d%H%M%S"):
    if timetag is None:
        return ""
    bj = dt.datetime.utcfromtimestamp(timetag / 1000.0) + dt.timedelta(hours=8)
    return bj.strftime(fmt)


def build_globals(engine):
    feed = engine.feed
    # NOTE: access engine.account dynamically (do NOT capture it here) -- on
    # checkpoint resume the engine swaps in a restored Account object.

    def _cur_bar(code):
        return feed.get_bar(code, engine.current_timetag)

    # --- account / position queries ------------------------------------

    def get_trade_detail_data(accountid, datatype, datakind, strategyname=""):
        engine.mark_holdings()                      # refresh盯市价
        kind = str(datakind).upper()
        if kind == "ACCOUNT":
            return engine.account.detail_account()
        if kind == "POSITION":
            return engine.account.detail_positions()
        return []

    # --- orders --------------------------------------------------------

    def passorder(opType, orderType, accountid, ordercode, prType,
                  modelprice, volume, *rest):
        # rest may be (tag, quickTrade, remark, C) or just (C,)
        code = ordercode
        shares = int(volume)
        tag = rest[0] if len(rest) >= 4 else ""
        bar = _cur_bar(code)
        if opType == OP_SELL:
            return engine.account.sell(code, shares, bar, engine.current_timetag, tag)
        if opType == OP_BUY:
            return engine.account.buy(code, shares, bar, engine.current_timetag, tag)
        return 0

    def order_target_value(code, target_value, priceType, C, accountid="", *a):
        bar = _cur_bar(code)
        if bar is None or bar.close <= 0:
            engine.account.rejects.append((engine.current_timetag, code, "no_bar"))
            return 0
        price = bar.close
        cur = engine.account.positions.get(code)
        cur_shares = cur.volume if cur else 0
        target_shares = int(target_value / price // 100) * 100
        delta = target_shares - cur_shares
        if delta > 0:
            return engine.account.buy(code, delta, bar, engine.current_timetag, "target_value")
        if delta < 0:
            return engine.account.sell(code, -delta, bar, engine.current_timetag, "target_value")
        return 0

    def order_target_percent(code, percent, priceType, C, accountid="", *a):
        return order_target_value(code, engine.account.total_value() * percent,
                                  priceType, C, accountid)

    def order_shares(code, shares, priceType="", price=-1, C=None, accountid="", *a):
        bar = _cur_bar(code)
        shares = int(shares)
        if shares > 0:
            return engine.account.buy(code, shares, bar, engine.current_timetag, "order_shares")
        if shares < 0:
            return engine.account.sell(code, -shares, bar, engine.current_timetag, "order_shares")
        return 0

    def get_market_data_ex(fields=[], stock_code=[], period="", start_time="",
                           end_time="", count=-1, dividend_type="",
                           fill_data=True, subscribe=True):
        return feed.get_market_data_ex(fields, stock_code, start_time, end_time, count)

    def get_bar(code, timetag_ms):
        # fast single-bar access (no pandas DataFrame)
        return feed.get_bar(code, timetag_ms)

    def get_bar_at(code, yyyymmdd, hhmmss):
        """Fast single-minute quote -> {field: value} dict (or None).

        Engine-only helper for the strategies' _quote hot path: it does the
        Beijing date/time -> timetag conversion and bar -> dict here (reusable),
        avoiding a pandas DataFrame.  QMT strategies fall back to
        get_market_data_ex when this global is absent.
        """
        b = feed.get_bar(code, beijing_to_timetag(yyyymmdd, hhmmss))
        if b is None:
            return None
        return {f: getattr(b, f) for f in _BAR_FIELDS}

    return {
        "get_trade_detail_data": get_trade_detail_data,
        "passorder": passorder,
        "order_target_value": order_target_value,
        "order_target_percent": order_target_percent,
        "order_shares": order_shares,
        "get_market_data_ex": get_market_data_ex,
        "get_bar": get_bar,
        "get_bar_at": get_bar_at,
        "timetag_to_datetime": timetag_to_datetime,
    }
