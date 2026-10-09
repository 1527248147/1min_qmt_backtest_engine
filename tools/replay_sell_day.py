#coding:utf-8
"""
Backtest TWAP sell vs the live fills of the same day, name by name.

    python tools/replay_sell_day.py --day 20261008 --exec <live exec_*.csv>

Seeds the engine account with exactly what the live exec file shows was sold
(code -> total filled shares), runs strategies/replay_twap_sell.py for that one
day on DuckDB minute bars, and compares the backtest's average sell price,
fill-time span and completion against the live ones.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from engine import Engine, DuckDBFeed, timetag_to_beijing
from engine.account import Position
import run_ml_backtest as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    ap.add_argument("--exec", required=True, help="live exec_*.csv of that day")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ex = pd.read_csv(a.exec, dtype={"t_place": str, "t_done": str})
    ex = ex[ex.qty_filled > 0].copy()
    ex["amt"] = ex.qty_filled * ex.price_filled
    live = ex.groupby("code").agg(live_sh=("qty_filled", "sum"), live_amt=("amt", "sum"),
                                  live_n=("qty_filled", "size"), live_first=("t_done", "min"),
                                  live_last=("t_done", "max"))
    live["live_px"] = live.live_amt / live.live_sh

    os.environ["REPLAY_DAY"] = a.day
    feed = DuckDBFeed(R.TS_DB, max_chunks=200, min_date=a.day, max_date=a.day)
    eng = Engine(feed=feed, driver_code=R.DRIVER, benchmark_code=R.BENCHMARK,
                 initial_capital=0.0, cost=R.COST, start_date=a.day, end_date=a.day,
                 corp_actions={})
    for code, row in live.iterrows():          # the live opening holding
        eng.account.positions[code] = Position(code, volume=int(row.live_sh),
                                               can_use=int(row.live_sh))
    eng.run(str(ROOT / "strategies" / "replay_twap_sell.py"))

    b = pd.DataFrame(eng.account.blotter,
                     columns=["timetag", "side", "code", "shares", "price", "fee", "tag"])
    b = b[b.side == "sell"].copy()
    b["hm"] = [timetag_to_beijing(int(t)).strftime("%H%M") for t in b.timetag]
    b["amt"] = b.shares * b.price
    bt = b.groupby("code").agg(bt_sh=("shares", "sum"), bt_amt=("amt", "sum"),
                               bt_n=("shares", "size"), bt_first=("hm", "min"), bt_last=("hm", "max"))
    bt["bt_px"] = bt.bt_amt / bt.bt_sh
    r = live.join(bt, how="left")
    r["bt_sh"] = r.bt_sh.fillna(0).astype(int)
    # sell: + = live sold HIGHER than the backtest (live better)
    r["live_vs_bt_bp"] = (r.live_px / r.bt_px - 1) * 1e4
    r["yuan"] = (r.live_px - r.bt_px) * r.live_sh
    pd.set_option("display.width", 220)
    cols = ["live_sh", "bt_sh", "live_px", "bt_px", "live_vs_bt_bp", "yuan",
            "live_n", "bt_n", "live_first", "bt_first", "live_last", "bt_last"]
    print(r[cols].round(4).sort_values("live_vs_bt_bp").to_string())
    done = r[r.bt_sh > 0]
    W = (done.live_px * done.live_sh).sum()
    wbp = ((done.live_vs_bt_bp * done.live_px * done.live_sh).sum() / W) if W else float("nan")
    print("\n%s: %d names | backtest sold %d / %d shares (%d names complete)"
          % (a.day, len(r), r.bt_sh.sum(), r.live_sh.sum(), int((r.bt_sh >= r.live_sh).sum())))
    print("live vs backtest, amount-weighted %+.1f bp (median %+.1f, live better on %d/%d), %+.0f yuan"
          % (wbp, done.live_vs_bt_bp.median(), int((done.live_vs_bt_bp > 0).sum()), len(done),
             done.yuan.sum()))
    print("orders: live %d, backtest %d" % (r.live_n.sum(), int(r.bt_n.fillna(0).sum())))
    if a.out:
        r.to_csv(a.out, encoding="utf-8-sig")
        print("->", a.out)


if __name__ == "__main__":
    main()
