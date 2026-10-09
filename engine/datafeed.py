#coding:utf-8
"""
QMT local .DAT minute-bar reader.

This reads exactly the bytes QMT itself reads, so the backtest is fed the same
minute data as the live QMT model engine.  The format was reverse-engineered
from QMT and confirmed against
``qmt_backtest/convert_external_1min_to_qmt_dat.py``:

    <QMT_ROOT>\\datadir\\SH\\60\\600000.DAT          (period folder "60" == 1m)
    <QMT_ROOT>\\datadir\\SZ\\60\\000001.DAT

Layout
------
* 8-byte header:  struct "<2i" == (-2, 2147483647)
* then N records of 64 bytes each, struct "<8iqiffiii":
      ts        int32   UTC seconds   (Beijing time minus 8h)
      open      int32   price * 1000
      high      int32   price * 1000
      low       int32   price * 1000
      close     int32   price * 1000
      _z0       int32   (always 0)
      volume    int32   shares        (external lots * 100)
      _c526     int32   (always 526)
      amount    int64   yuan
      status    int32   14=09:30  13=continuous  18=14:58/14:59 auction  15=15:00
      _f1       float32 (1.0)
      _f2       float32 (1.0)
      preclose  int32   price * 1000
      _z1       int32   (0)
      _c32763   int32   (32763)

``timetag`` everywhere in this engine is **milliseconds** (= ts * 1000), the same
unit QMT's ``get_bar_timetag`` returns.  Conversion to Beijing wall-clock matches
the strategy's own ``_bar_datetime``::

    datetime.utcfromtimestamp(timetag / 1000) + timedelta(hours=8)
"""

from __future__ import annotations

import struct
import datetime as dt
from pathlib import Path

import pandas as pd

HEADER_SIZE = 8
RECORD_SIZE = 64
_REC = struct.Struct("<8iqiffiii")
_TS = struct.Struct("<i")

# field name -> position in the unpacked record tuple
_FIELD_POS = {
    "time": 0,      # raw ts (seconds); exposed as timetag(ms) on demand
    "open": 1,
    "high": 2,
    "low": 3,
    "close": 4,
    "volume": 6,
    "amount": 8,
    "status": 9,
    "preclose": 12,
}
_PRICE_FIELDS = {"open", "high", "low", "close", "preclose"}


def timetag_to_beijing(timetag_ms: int) -> dt.datetime:
    """timetag (ms, UTC based) -> naive Beijing datetime, matching the strategy."""
    return dt.datetime.utcfromtimestamp(timetag_ms / 1000.0) + dt.timedelta(hours=8)


def beijing_to_timetag(date_text: str, hhmmss: str) -> int:
    """('20180103','093000') -> timetag in ms (UTC based)."""
    bj = dt.datetime.strptime(date_text + hhmmss, "%Y%m%d%H%M%S")
    utc = bj - dt.timedelta(hours=8)
    return int(utc.replace(tzinfo=dt.timezone.utc).timestamp()) * 1000


class Bar:
    __slots__ = ("timetag", "open", "high", "low", "close", "volume", "amount",
                 "status", "preclose")

    def __init__(self, rec_tuple):
        self.timetag = rec_tuple[0] * 1000            # ms
        self.open = rec_tuple[1] / 1000.0
        self.high = rec_tuple[2] / 1000.0
        self.low = rec_tuple[3] / 1000.0
        self.close = rec_tuple[4] / 1000.0
        self.volume = rec_tuple[6]                    # shares
        self.amount = rec_tuple[8]                    # yuan
        self.status = rec_tuple[9]
        self.preclose = rec_tuple[12] / 1000.0

    def __repr__(self):
        return (f"Bar({timetag_to_beijing(self.timetag):%Y-%m-%d %H:%M} "
                f"o={self.open} h={self.high} l={self.low} c={self.close} "
                f"v={self.volume})")


class DataFeed:
    """Reads QMT .DAT minute files on demand and caches per-code bars in memory."""

    def __init__(self, datadir):
        # datadir e.g. r"...\迅投...\datadir"  (it holds SH\60, SZ\60, BJ\60)
        self.datadir = Path(datadir)
        self._cache: dict[str, dict[int, Bar]] = {}   # code -> {timetag_ms: Bar}

    # ----- path / availability -----------------------------------------

    def _dat_path(self, code: str) -> Path:
        # code "600000.SH" -> datadir\SH\60\600000.DAT
        stk, market = code.split(".")
        return self.datadir / market.upper() / "60" / f"{stk}.DAT"

    def has_code(self, code: str) -> bool:
        return self._dat_path(code).exists()

    # ----- raw load (cached) -------------------------------------------

    def _load(self, code: str) -> dict[int, Bar]:
        cached = self._cache.get(code)
        if cached is not None:
            return cached
        path = self._dat_path(code)
        bars: dict[int, Bar] = {}
        if path.exists():
            data = path.read_bytes()
            n = (len(data) - HEADER_SIZE) // RECORD_SIZE
            off = HEADER_SIZE
            for _ in range(n):
                rec = _REC.unpack_from(data, off)
                bar = Bar(rec)
                bars[bar.timetag] = bar
                off += RECORD_SIZE
        self._cache[code] = bars
        return bars

    # ----- public access -----------------------------------------------

    def get_bar(self, code: str, timetag_ms: int):
        """Single bar for code at an exact timetag, or None if absent."""
        return self._load(code).get(timetag_ms)

    def list_timetags(self, code: str):
        """Sorted list of all bar timetags (ms) for a code -- used as driver."""
        return sorted(self._load(code).keys())

    def get_market_data_ex(self, fields, stock_code, start_time="", end_time="",
                           count=-1):
        """
        Mirror of QMT get_market_data_ex.

        fields:     list like ["open","close","volume","amount"]; [] -> all
        stock_code: list of codes "600000.SH"
        start_time/end_time: "" or "YYYYMMDDHHMMSS" (Beijing); inclusive range
        count:      -1 all in range; >0 keep last `count` rows
        returns:    {code: DataFrame indexed by stime "YYYYMMDDHHMMSS"}
        """
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
            bars = self._load(code)
            tts = sorted(bars)
            if start_ms is not None:
                tts = [t for t in tts if t >= start_ms]
            if end_ms is not None:
                tts = [t for t in tts if t <= end_ms]
            if count is not None and count > 0:
                tts = tts[-count:]

            index = []
            rows = []
            for t in tts:
                b = bars[t]
                index.append(timetag_to_beijing(t).strftime("%Y%m%d%H%M%S"))
                row = {}
                for f in fields:
                    if f == "time":
                        row["time"] = t
                    else:
                        row[f] = getattr(b, f, float("nan"))
                rows.append(row)
            df = pd.DataFrame(rows, index=index, columns=list(fields))
            df.index.name = "stime"
            out[code] = df
        return out
