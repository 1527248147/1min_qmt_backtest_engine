#coding:utf-8
"""
Backtest runner -- the StrategyLoader / run_bar equivalent.

Mirrors QMT's loop (xtquant/qmttools/stgframe.py::run_bar):

    for i in range(len(timelist)):
        C.barpos = i
        handlebar(C)

with three additions QMT does inside its closed engine:
  * the timeline ("timelist") comes from a driver symbol's .DAT bars,
  * T+1 settlement runs at each new trading day,
  * end-of-day equity (NAV) is recorded for a performance report.
"""

from __future__ import annotations

import os
import pickle
import time as _time
import types
import datetime as dt

from .datafeed import DataFeed, timetag_to_beijing
from .account import Account, CostModel, load_st_ranges
from .context import ContextInfo
from .api import build_globals


class Engine:
    def __init__(self, datadir=None, driver_code="000001.SZ",
                 initial_capital=1_000_000.0, cost=None,
                 start_date=None, end_date=None, benchmark_code=None,
                 feed=None, corp_actions=None):
        # feed: pass a ParquetFeed for full-history backtests; otherwise a
        # .DAT DataFeed is built from `datadir` (the live QMT data path).
        self.feed = feed if feed is not None else DataFeed(datadir)
        # corp_actions: {code: {ex_date_int: (stk_div, cash_div)}} for 送转/分红
        self.corp_actions = corp_actions or {}
        # ST ranges make the +-5% limit apply on the dates a name actually
        # carried the ST tag. Empty file -> old behaviour (board limit for all).
        self.account = Account(cash=initial_capital, cost=cost or CostModel(),
                               st_ranges=load_st_ranges())
        self.driver_code = driver_code
        self.benchmark_code = benchmark_code or driver_code
        self.initial_capital = initial_capital
        self.start_date = start_date          # "YYYYMMDD" or None
        self.end_date = end_date
        self.current_timetag = None
        self.universe = []

        self.C = ContextInfo(self)
        self.C.capital = initial_capital
        self.C.asset = initial_capital

        # outputs
        self.equity = []     # (date, total_value, cash, stock_value, bench_close)
        # daily holdings snapshot for the 后复权 TWRR equity curve:
        #   (date, cash, {code: (shares, raw_close)})
        self.daily_holdings = []
        self._bench_base = None
        self._last_mark_key = None

    # ----- checkpoint / resume ------------------------------------------

    def _save_checkpoint(self, path, resume_i, cur_day, prev_tt):
        """Pickle enough state to resume the bar loop at `resume_i` after a crash.

        The strategy's ContextInfo (C) holds all its state as plain attributes;
        we pickle them minus the engine ref, the (rebuilt) timelist and any bound
        methods.  load_strategy re-binds methods + re-injects globals on resume.
        """
        C = self.C
        c_state = {k: v for k, v in C.__dict__.items()
                   if k not in ("_engine", "timelist") and not callable(v)}
        data = {"resume_i": resume_i, "cur_day": cur_day, "prev_tt": prev_tt,
                "account": self.account, "c_state": c_state,
                "equity": self.equity, "daily_holdings": self.daily_holdings}
        # A checkpoint is a resume hint, NOT part of the result. Failing to write
        # one must never kill a run that is otherwise fine.
        #
        # On Windows os.replace() raises WinError 32 whenever anything else has
        # the target open for even a moment -- Defender scanning the file just
        # written, the indexer, a sync client. On 2026-08-06 that aborted two of
        # four backtests at ~80% of a 2048-day run, each on its OWN checkpoint
        # file, so it was not the cross-basket collision fixed separately: the
        # rename simply lost a race with a scanner.
        #
        # Retry a few times, then give up quietly and keep going. Losing a
        # checkpoint costs a restart from bar 0 if the run later crashes; raising
        # here costs the entire run, every time.
        tmp = path + ".tmp"
        try:
            with open(tmp, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            for attempt in range(6):
                try:
                    os.replace(tmp, path)
                    return
                except OSError:
                    if attempt == 5:
                        raise
                    _time.sleep(0.25)
        except Exception as e:
            if not getattr(self, "_ckpt_warned", False):
                self._ckpt_warned = True
                print(f"  checkpoint save failed ({e}); continuing without it",
                      flush=True)
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass

    def _load_checkpoint(self, path):
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  checkpoint load failed ({e}); starting fresh", flush=True)
            return None

    # ----- helpers ------------------------------------------------------

    def mark_holdings(self, force=False):
        """Refresh last_price of held positions from bars at the current bar.

        Memoized per (timetag, position-set): get_trade_detail_data calls this on
        every query, but the marks only change when the bar advances, so we skip
        redundant re-marking within the same bar.
        """
        key = (self.current_timetag, len(self.account.positions))
        if not force and key == self._last_mark_key:
            return
        for code in list(self.account.positions):
            bar = self.feed.get_bar(code, self.current_timetag)
            if bar is not None and bar.close > 0:
                self.account.mark(code, bar.close)
        self._last_mark_key = key

    # ----- strategy loading (stgentry.run_file equivalent) --------------

    def load_strategy(self, script_path):
        src = open(script_path, "rb").read()           # respects #coding: header
        code = compile(src, script_path, "exec")
        ns = {}
        ns.update(build_globals(self))                 # inject QMT globals
        exec(code, ns, ns)

        C = self.C
        for name in ("init", "after_init", "handlebar", "stop"):
            fn = ns.get(name)
            if fn:
                setattr(C, name, types.MethodType(fn, C))
        self._strategy_ns = ns
        return self

    # ----- main loop (stgframe.run_bar equivalent) ----------------------

    def _build_timelist(self):
        tts = self.feed.list_timetags(self.driver_code)
        if not tts:
            raise RuntimeError(
                f"driver {self.driver_code} has no .DAT bars under {self.feed.datadir}")
        if self.start_date or self.end_date:
            def keep(t):
                d = timetag_to_beijing(t).strftime("%Y%m%d")
                if self.start_date and d < self.start_date:
                    return False
                if self.end_date and d > self.end_date:
                    return False
                return True
            tts = [t for t in tts if keep(t)]
        return tts

    def run(self, script_path, progress_every=0, checkpoint_path=None,
            checkpoint_every=10000):
        self.load_strategy(script_path)        # always: binds methods, injects globals
        C = self.C
        C.timelist = self._build_timelist()
        C.stock_code = self.driver_code
        C.stockcode, C.market = self.driver_code.split(".")
        n = len(C.timelist)

        ck = self._load_checkpoint(checkpoint_path)
        if ck:
            self.account = ck["account"]
            for k, v in ck["c_state"].items():
                setattr(C, k, v)
            C._engine = self
            self.equity = ck["equity"]
            self.daily_holdings = ck["daily_holdings"]
            cur_day, prev_tt, start_i = ck["cur_day"], ck["prev_tt"], ck["resume_i"]
            print(f"  resumed from checkpoint at bar {start_i}/{n}", flush=True)
        else:
            if hasattr(C, "init"):
                C.init()
            if hasattr(C, "after_init"):
                C.after_init()
            cur_day, prev_tt, start_i = None, None, 0

        for i in range(start_i, n):
            tt = C.timelist[i]
            day = timetag_to_beijing(tt).strftime("%Y%m%d")

            if day != cur_day:
                if cur_day is not None:
                    # record the finishing day at ITS last bar (prev_tt), not the
                    # new day's first bar -- otherwise NAV is marked at next-day open
                    self.current_timetag = prev_tt
                    self._record_equity(cur_day)
                self.account.settle_new_day()           # T+1: 昨仓可卖
                self.account.apply_corporate_actions(int(day), self.corp_actions)  # 送转/分红
                cur_day = day

            self.current_timetag = tt
            C.barpos = i
            C.handlebar()
            prev_tt = tt

            if progress_every and (i % progress_every == 0 or i == n - 1):
                print(f"  bar {i+1}/{n}  {day}  nav={self.account.total_value():,.0f}",
                      flush=True)
            if checkpoint_path and i > start_i and i % checkpoint_every == 0:
                self._save_checkpoint(checkpoint_path, i + 1, cur_day, prev_tt)

        if cur_day is not None:
            self.current_timetag = prev_tt
            self._record_equity(cur_day)
        if hasattr(C, "stop"):
            C.stop()
        if checkpoint_path and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)       # finished cleanly -> drop checkpoint
        return self.report()

    def _record_equity(self, day):
        self.mark_holdings()
        snap = {code: (p.volume, p.last_price)
                for code, p in self.account.positions.items() if p.volume > 0}
        self.daily_holdings.append((day, self.account.cash, snap))
        bench = self.feed.get_bar(self.benchmark_code, self.current_timetag)
        bench_close = bench.close if bench else (
            self.equity[-1][4] if self.equity else None)
        self.equity.append((
            day,
            self.account.total_value(),
            self.account.cash,
            self.account.stock_value(),
            bench_close,
        ))

    # ----- reporting ----------------------------------------------------

    def report(self):
        import pandas as pd
        df = pd.DataFrame(self.equity,
                          columns=["date", "nav", "cash", "stock_value", "bench"])
        if df.empty:
            return df
        df["ret"] = df["nav"].pct_change().fillna(0.0)
        df["nav_norm"] = df["nav"] / self.initial_capital
        if df["bench"].notna().any():
            base = df["bench"].dropna().iloc[0]
            df["bench_norm"] = df["bench"] / base
        return df
