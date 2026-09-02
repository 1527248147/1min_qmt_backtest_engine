#coding:utf-8
"""
ParquetFeed -- minute data source backed by the full-history per-stock parquet
store at C:\\AI_STOCK\\dataset\\1min_parquet\\<code>.parquet (volume in SHARES).

Same interface as datafeed.DataFeed (get_bar / list_timetags / get_market_data_ex)
so the engine, api and runner use it unchanged.  Use this for full-history
backtests; keep the .DAT DataFeed for the small live-traded subset inside QMT.

Memory: each code is cached as compact numpy arrays (struct-of-arrays); an LRU
keeps at most ``max_codes`` codes resident (the driver stays hot because it is
read every bar).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from .datafeed import timetag_to_beijing, beijing_to_timetag

# cached (hot) fields kept compact; amount is loaded on demand only, not cached
_FIELDS = ("open", "high", "low", "close", "volume", "preclose")
_DTYPE = {"open": "float32", "high": "float32", "low": "float32",
          "close": "float32", "volume": "int64", "preclose": "float32"}


class BarView:
    __slots__ = ("timetag", "open", "high", "low", "close", "volume", "amount", "preclose")

    def __init__(self, tt, o, h, l, c, v, a, pc):
        self.timetag = int(tt)
        self.open = o; self.high = h; self.low = l; self.close = c
        self.volume = v; self.amount = a; self.preclose = pc

    def __repr__(self):
        return (f"Bar({timetag_to_beijing(self.timetag):%Y-%m-%d %H:%M} "
                f"o={self.open} c={self.close} v={self.volume})")


class ParquetFeed:
    def __init__(self, parquet_dir, max_codes=400, adjust=False, factors=None):
        self.dir = Path(parquet_dir)
        self.max_codes = max_codes
        # adjust="hfq": multiply price fields by the daily adj_factor (normalised
        # to the code's first bar) so the engine runs on 后复权 prices. Used as an
        # independent cross-check of the raw+corporate-action run. factors:
        # {code: {trade_date_int: adj_factor}}.
        self.adjust = adjust
        self.factors = factors or {}
        self._cache: "OrderedDict[str, dict]" = OrderedDict()

    def _path(self, code):
        return self.dir / f"{code}.parquet"

    def has_code(self, code):
        return self._path(code).exists()

    def _load(self, code):
        hit = self._cache.get(code)
        if hit is not None:
            self._cache.move_to_end(code)
            return hit
        path = self._path(code)
        if not path.exists():
            store = {"tt": np.empty(0, dtype="int64")}
        else:
            cols = ["timetag", *_FIELDS] + (["trade_date"] if self.adjust else [])
            df = pd.read_parquet(path, columns=cols)
            # parquet is written timetag-sorted by export_1min_to_parquet.py, so
            # skip the per-load sort (was ~30% of runtime); only sort if a file is
            # unexpectedly out of order (cheap O(n) monotonic check).
            _tt = df["timetag"].to_numpy()
            if _tt.size and not (np.diff(_tt) >= 0).all():
                df = df.sort_values("timetag")
            if self.adjust:
                fac_map = self.factors.get(code)
                if fac_map:
                    fseries = pd.Series(fac_map).sort_index()
                    td = df["trade_date"].to_numpy()
                    idx = np.searchsorted(fseries.index.to_numpy(), td, side="right") - 1
                    idx = np.clip(idx, 0, len(fseries) - 1)
                    fac = fseries.to_numpy()[idx]
                    fac = fac / fac[0]                       # normalise to first bar
                    for pf in ("open", "high", "low", "close", "preclose"):
                        df[pf] = df[pf].to_numpy() * fac
            store = {"tt": df["timetag"].to_numpy(dtype="int64")}
            for f in _FIELDS:
                store[f] = df[f].to_numpy(dtype=_DTYPE[f])
        self._cache[code] = store
        self._cache.move_to_end(code)
        while len(self._cache) > self.max_codes:
            self._cache.popitem(last=False)
        return store

    # ----- public API (mirrors datafeed.DataFeed) ----------------------

    def get_bar(self, code, timetag_ms):
        s = self._load(code)
        tt = s["tt"]
        if tt.size == 0:
            return None
        i = np.searchsorted(tt, timetag_ms)
        if i >= tt.size or tt[i] != timetag_ms:
            return None
        return BarView(timetag_ms, float(s["open"][i]), float(s["high"][i]),
                       float(s["low"][i]), float(s["close"][i]), int(s["volume"][i]),
                       0.0, float(s["preclose"][i]))   # amount not cached (unused by matcher)

    def list_timetags(self, code):
        return self._load(code)["tt"].tolist()

    def daily_close_series(self, code):
        """Raw daily close per trade_date (int YYYYMMDD) -> dict, last bar of each day."""
        path = self._path(code)
        if not path.exists():
            return {}
        df = pd.read_parquet(path, columns=["trade_date", "close"])
        last = df.groupby("trade_date")["close"].last()
        return {int(k): float(v) for k, v in last.items()}

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
            s = self._load(code)
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
