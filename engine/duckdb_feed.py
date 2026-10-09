#coding:utf-8
"""
DuckDBFeed -- minute data source backed by the unified vendor minute table in
tushare.duckdb (``minute_l1.bars``; ``market.minute_1m`` is the A-share view of
it). Replaces ParquetFeed: the per-stock parquet store it read was folded into
DuckDB on 2026-10-03 and the parquet files deleted.

Same public interface as ParquetFeed / datafeed.DataFeed (get_bar /
list_timetags / get_market_data_ex / daily_close_series / has_code), so the
engine, api and runner use it unchanged.

Column mapping (the table already uses the engine's conventions):
  timetag   BIGINT ms, true UTC epoch -- 09:30 Beijing on 2024-01-02 is
            1704159000000, identical to the old parquet and to QMT .DAT
  volume    DOUBLE shares (= volume_lots * 100); rounded to int64 here, as
            the parquet exporter did
  preclose  NOT stored. Derived exactly as export_1min_to_parquet.py did it:
            the previous trading day's last close, falling back to the bar's
            own close only when no earlier day exists at all.

CACHE BY (code, year), NOT BY code. ParquetFeed cached a code's whole
history, ~19 MB for 2018..2026; a top-10% basket touches ~1,000 codes, so
800 resident codes was ~15 GB per process. Every request the engine and the
TWAP strategy make falls inside one trading day, so a year chunk (~2 MB) is
enough, and ``max_chunks`` bounds memory at roughly max_chunks * 2.2 MB.
The cache must still hold the working set -- every held name is read every
bar -- or LRU thrashes; size it above (held + candidates) for the basket.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from .datafeed import timetag_to_beijing, beijing_to_timetag
from .parquet_feed import BarView, _FIELDS, _DTYPE

TABLE = "minute_l1.bars"


def _months_before(yyyymmdd, months):
    y, m = divmod(int(yyyymmdd) // 100, 100)
    m -= months
    while m < 1:
        m += 12
        y -= 1
    return y * 10000 + m * 100 + 1


def _year_of(tt_ms):
    return timetag_to_beijing(int(tt_ms)).year


class DuckDBFeed:
    def __init__(self, db_path, max_chunks=1500, adjust=False, factors=None,
                 min_date=None, max_date=None, max_codes=None):
        import duckdb
        self.db_path = str(db_path)
        self.con = duckdb.connect(self.db_path, read_only=True)
        # Several baskets run side by side; DuckDB's default of one thread per
        # core in every process oversubscribes the CPU. FEED_DUCKDB_THREADS
        # overrides.
        import os as _os
        self.con.execute("SET threads TO %d" % int(_os.environ.get("FEED_DUCKDB_THREADS", "4")))
        # max_codes: accepted for call compatibility with ParquetFeed; a code
        # spans several year chunks, so it is converted, not used as-is.
        self.max_chunks = int(max_chunks if max_codes is None else max(max_chunks, max_codes))
        self.adjust = adjust
        self.factors = factors or {}
        self.min_date = int(min_date) if min_date else None
        self.max_date = int(max_date) if max_date else None
        self._cache: "OrderedDict[tuple, dict]" = OrderedDict()
        self._codes = None
        self.datadir = self.db_path            # runner's error message names it

    # ----- loading -------------------------------------------------------

    def has_code(self, code):
        if self._codes is None:
            self._codes = {r[0] for r in self.con.execute(
                f"SELECT DISTINCT ts_code FROM {TABLE} WHERE share_class = 'A'").fetchall()}
        return code in self._codes

    def _years(self):
        lo = (self.min_date or 20000101) // 10000
        hi = (self.max_date or 20991231) // 10000
        return lo, hi

    def _chunk(self, code, year):
        key = (code, year)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        lo = max(year * 10000 + 101, self.min_date or 0)
        hi = min(year * 10000 + 1231, self.max_date or 99991231)
        store = {"tt": np.empty(0, dtype="int64")}
        if lo <= hi:
            # one extra trading day before lo, for its preclose
            # Read from three months before lo so the first day kept has a real
            # preclose (the previous trading day's last close). A separate
            # "last day before lo" lookup was tried twice and dropped: max()
            # hits a DuckDB 1.5.3 INTERNAL error on codes with no earlier row,
            # and ORDER BY ... LIMIT 1 is unbounded -- up to 2.6 s per code,
            # which stalled a top-10% basket loading 1,100 codes on day one.
            # A code suspended for the whole three months falls back to its own
            # close on that first day, as the parquet exporter did on day one.
            q_lo = _months_before(lo, 3)
            a = self.con.execute(
                f"SELECT timetag, trade_date, open, high, low, close, volume FROM {TABLE} "
                "WHERE ts_code = ? AND share_class = 'A' AND trade_date BETWEEN ? AND ? "
                "ORDER BY timetag", [code, q_lo, hi]).fetchnumpy()
            tt = np.asarray(a["timetag"], dtype="int64")
            if tt.size:
                td = np.asarray(a["trade_date"], dtype="int64")
                close = np.asarray(a["close"], dtype="float64")
                new_day = np.r_[True, td[1:] != td[:-1]]
                starts = np.flatnonzero(new_day)
                ends = np.r_[starts[1:], td.size] - 1
                prev_last = np.r_[np.nan, close[ends][:-1]]
                preclose = prev_last[np.cumsum(new_day) - 1]
                preclose = np.where(np.isnan(preclose), close, preclose)
                cols = {"open": np.asarray(a["open"], dtype="float64"),
                        "high": np.asarray(a["high"], dtype="float64"),
                        "low": np.asarray(a["low"], dtype="float64"),
                        "close": close, "preclose": preclose,
                        "volume": np.rint(np.nan_to_num(np.asarray(a["volume"], dtype="float64")))}
                if self.adjust:
                    fac_map = self.factors.get(code)
                    if fac_map:
                        fs = pd.Series(fac_map).sort_index()
                        idx = np.clip(np.searchsorted(fs.index.to_numpy(), td, side="right") - 1,
                                      0, len(fs) - 1)
                        # normalised to the code's FIRST factor, so every year
                        # chunk shares one scale (ParquetFeed: first bar loaded)
                        fac = fs.to_numpy()[idx] / fs.to_numpy()[0]
                        for pf in ("open", "high", "low", "close", "preclose"):
                            cols[pf] = cols[pf] * fac
                keep = td >= lo
                store = {"tt": tt[keep]}
                for f in _FIELDS:
                    store[f] = cols[f][keep].astype(_DTYPE[f])
        self._cache[key] = store
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_chunks:
            self._cache.popitem(last=False)
        return store

    def _span(self, code, start_ms, end_ms):
        """Concatenated arrays for [start_ms, end_ms] (None = window edge)."""
        y0, y1 = self._years()
        if start_ms is not None:
            y0 = max(y0, _year_of(start_ms))
        if end_ms is not None:
            y1 = min(y1, _year_of(end_ms))
        parts = [self._chunk(code, y) for y in range(y0, y1 + 1)]
        parts = [p for p in parts if p["tt"].size]
        if not parts:
            return {"tt": np.empty(0, dtype="int64"),
                    **{f: np.empty(0, dtype=_DTYPE[f]) for f in _FIELDS}}
        if len(parts) == 1:
            return parts[0]
        return {k: np.concatenate([p[k] for p in parts]) for k in ("tt",) + _FIELDS}

    # ----- public API (mirrors datafeed.DataFeed) ----------------------

    def get_bar(self, code, timetag_ms):
        s = self._chunk(code, _year_of(timetag_ms))
        tt = s["tt"]
        if tt.size == 0:
            return None
        i = np.searchsorted(tt, timetag_ms)
        if i >= tt.size or tt[i] != timetag_ms:
            return None
        return BarView(timetag_ms, float(s["open"][i]), float(s["high"][i]),
                       float(s["low"][i]), float(s["close"][i]), int(s["volume"][i]),
                       0.0, float(s["preclose"][i]))

    def list_timetags(self, code):
        return self._span(code, None, None)["tt"].tolist()

    def daily_close_series(self, code):
        """Raw daily close per trade_date (int YYYYMMDD), last bar of each day."""
        rows = self.con.execute(
            f"SELECT trade_date, arg_max(close, timetag) FROM {TABLE} "
            "WHERE ts_code = ? AND share_class = 'A' AND trade_date BETWEEN ? AND ? "
            "GROUP BY 1 ORDER BY 1",
            [code, self.min_date or 0, self.max_date or 99991231]).fetchall()
        return {int(d): float(c) for d, c in rows}

    def get_market_data_ex(self, fields, stock_code, start_time="", end_time="", count=-1):
        if isinstance(stock_code, str):
            stock_code = [stock_code]
        if not fields:
            fields = ["open", "high", "low", "close", "volume", "amount"]
        start_ms = (beijing_to_timetag(start_time[:8], start_time[8:14])
                    if len(start_time) >= 14 else None)
        end_ms = (beijing_to_timetag(end_time[:8], end_time[8:14])
                  if len(end_time) >= 14 else None)
        out = {}
        for code in stock_code:
            s = self._span(code, start_ms, end_ms)
            tt = s["tt"]
            lo = 0 if start_ms is None else int(np.searchsorted(tt, start_ms, "left"))
            hi = tt.size if end_ms is None else int(np.searchsorted(tt, end_ms, "right"))
            sl = slice(lo, hi)
            idx = [timetag_to_beijing(int(t)).strftime("%Y%m%d%H%M%S") for t in tt[sl]]
            data = {}
            for f in fields:
                if f == "time":
                    data["time"] = tt[sl]
                elif f in _FIELDS:
                    data[f] = s[f][sl]
                else:
                    data[f] = np.full(hi - lo, np.nan)
            df = pd.DataFrame(data, index=idx, columns=list(fields))
            if count and count > 0:
                df = df.iloc[-count:]
            df.index.name = "stime"
            out[code] = df
        return out
