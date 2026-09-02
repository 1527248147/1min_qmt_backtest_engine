#coding:utf-8
"""
ContextInfo -- the `C` object passed to init(C)/handlebar(C).

Modeled on QMT's GUI model-backtest ContextInfo (the `_PyContextInfo.py`
flavour the ML strategy targets), not the lighter qmttools one.  Only the
members the strategy actually touches are implemented; everything else can be
set as a plain attribute by the strategy (e.g. C.pending_sells).
"""

from __future__ import annotations


class ContextInfo:
    def __init__(self, engine):
        self._engine = engine

        # bar frame
        self.barpos = -1
        self.timelist = []          # list of timetags (ms)

        # quote / run config
        self.stock_code = ""
        self.stockcode = ""
        self.market = ""
        self.period = "1m"
        self.dividend_type = "none"
        self.start = ""
        self.end = ""
        self.start_time = ""
        self.end_time = ""
        self.benchmark = ""

        # backtest config
        self.asset = 1000000.0
        self.capital = 1000000.0

        # account binding (strategy usually sets C.account_id itself)
        self.account_id = "testS"

    # ----- bar helpers (QMT API) ---------------------------------------

    def get_bar_timetag(self, barpos=None):
        try:
            i = self.barpos if barpos is None else barpos
            return self.timelist[i]
        except Exception:
            return None

    def is_last_bar(self):
        return self.barpos >= len(self.timelist) - 1

    # ----- quote (QMT API) ---------------------------------------------

    def get_market_data_ex(self, fields=[], stock_code=[], period="",
                           start_time="", end_time="", count=-1,
                           dividend_type="", fill_data=True, subscribe=True):
        if not stock_code:
            stock_code = [self.stock_code]
        return self._engine.feed.get_market_data_ex(
            fields, stock_code, start_time, end_time, count)

    # ----- universe / misc (no-ops sufficient for backtest) -------------

    def set_universe(self, universe):
        self._engine.universe = list(universe)

    def get_universe(self):
        return getattr(self._engine, "universe", [])

    def set_account(self, account_id, account_type=""):
        self.account_id = account_id

    def paint(self, *a, **k):
        return None
