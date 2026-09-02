#coding:utf-8
"""
Entry point: run an ML stock-selection strategy on the QMT-faithful 1m engine.

Thin wiring only -- the engine (matching, corporate actions, performance
analytics) lives in the `engine` package and is reusable.

Usage:
    python run_ml_backtest.py --strategy qmt_ml_signal_model_backtest_v2.py
    python run_ml_backtest.py --strategy ... --start 20180101 --end 20181231
    python run_ml_backtest.py --strategy ... --adjust hfq   # 后复权价执行交叉验证
Outputs are suffixed with the strategy stem so runs don't clobber each other.
"""

import argparse
import os as _os
import hashlib as _hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from engine import (Engine, CostModel, ParquetFeed, print_metrics,
                    hfq_twrr_curve, load_corp_actions, load_adj_factors)

PARQUET_DIR = r"C:\AI_STOCK\dataset\1min_parquet"   # built by export_1min_to_parquet.py
TS_DB = r"C:\AI_STOCK\dataset\duckdb\tushare.duckdb"
STRATEGY_DIR = Path(__file__).resolve().parent / "strategies"

DRIVER = "000001.SZ"      # provides the 241-bar/day timeline, like QMT main chart
BENCHMARK = "000001.SZ"   # driver as benchmark (sparse index .DAT unavailable)
INITIAL_CAPITAL = 10_000_000.0

# A-share cost model: 佣金万0.8(无最低佣金, 按用户实际券商), 印花税现实时变
# (2023-08-28起0.001->0.0005), 过户费万0.1双边, 规费万0.625双边.
# max_vol_rate=1.0 (Account 默认).
#
# 规费在 2026-09-02 之前完全没有计入. 买卖各万0.625, 一次完整换手 1.25bp --
# 和佣金双边的 1.6bp 同一量级. 月度换仓每年 12 次完整换手, 约 15bp 的年化
# 拖累. 在此之前的所有回测结果都因此偏乐观这么多.
COST = CostModel(commission_rate=0.00008, min_commission=0.0,
                 transfer_fee=0.00001, regulatory_fee=0.0000625,
                 slippage_bps=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20180101")
    ap.add_argument("--end", default=None)
    ap.add_argument("--strategy", default="qmt_ml_signal_model_backtest_v2.py")
    ap.add_argument("--adjust", default="", choices=["", "hfq"],
                    help="hfq = run execution on 后复权 prices (independent cross-check)")
    ap.add_argument("--capital", type=float, default=INITIAL_CAPITAL,
                    help="initial account capital (default 10,000,000)")
    # Execution-cost knobs. Defaults reproduce the original run exactly, so an
    # existing command line gives the same numbers as before.
    ap.add_argument("--slippage-bps", type=float, default=None,
                    help="crossing cost in bps; 6.2 approximates one tick of "
                         "half-spread on this universe (median price 6.96)")
    ap.add_argument("--per-order-fee", type=float, default=None,
                    help="fixed CNY per order, e.g. 0.1 flow fee (0.2 to "
                         "approximate live cancel-and-requote doubling it)")
    ap.add_argument("--commission-rate", type=float, default=None,
                    help="override the 0.00008 (wan 0.8) commission rate")
    ap.add_argument("--regulatory-fee", type=float, default=None,
                    help="override the 0.0000625 (wan 0.625) regulatory fee "
                         "charged on both sides; 0 reproduces every run made "
                         "before 2026-09-02, when it was not modelled at all")
    ap.add_argument("pos", nargs="*", help=argparse.SUPPRESS)   # legacy: START END
    args = ap.parse_args()
    if args.slippage_bps is not None:
        COST.slippage_bps = args.slippage_bps
    if args.per_order_fee is not None:
        COST.per_order_fee = args.per_order_fee
    if args.commission_rate is not None:
        COST.commission_rate = args.commission_rate
    if args.regulatory_fee is not None:
        COST.regulatory_fee = args.regulatory_fee
    print("COST: commission %.5f  regulatory %.7f  slippage %.1fbp"
          "  per-order %.2f CNY"
          % (COST.commission_rate, COST.regulatory_fee, COST.slippage_bps,
             COST.per_order_fee))
    start, end = args.start, args.end
    if args.pos:
        start = args.pos[0]
        end = args.pos[1] if len(args.pos) > 1 else end

    strategy = str(STRATEGY_DIR / args.strategy)
    stem = Path(args.strategy).stem

    if args.adjust == "hfq":
        # cross-check: run execution on 后复权 prices (factor bakes in 送转/分红),
        # so engine corporate-action handling is OFF here.
        print("loading adj_factors for 后复权 run ...")
        factors = load_adj_factors(TS_DB)
        feed = ParquetFeed(PARQUET_DIR, max_codes=800, adjust="hfq", factors=factors)
        corp = {}
        stem += "_hfqrun"
        print(f"factors for {len(factors)} codes; CA handling OFF (in price)")
    else:
        print("loading corporate actions (送转/分红) ...")
        corp = load_corp_actions(TS_DB)
        feed = ParquetFeed(PARQUET_DIR, max_codes=800)
        print(f"corp actions for {len(corp)} codes")

    eng = Engine(feed=feed, driver_code=DRIVER, benchmark_code=BENCHMARK,
                 initial_capital=args.capital, cost=COST,
                 start_date=start, end_date=end, corp_actions=corp)
    out = Path(__file__).resolve().parent / "data"
    out.mkdir(exist_ok=True)
    # The checkpoint key MUST cover everything that changes the result, not just
    # the strategy filename. All four baskets (top20 / 2.5% / 5% / 10%) run the
    # SAME strategy file and differ only by COMBO_PLAN_CSV, so a single
    # _checkpoint_qmt_combo_top20_twap.pkl was shared by all of them. A run
    # killed part-way left that file behind and the NEXT run silently resumed
    # from it -- inheriting the other basket's positions, cash and strategy
    # state. Observed 2026-08-06: a top20 run resumed from a top5% checkpoint
    # and reported 4.47M fills (top5% scale) instead of ~0.8M.
    _key = "|".join([
        stem,
        _os.environ.get("COMBO_PLAN_CSV", "default"),
        str(args.capital), str(args.slippage_bps),
        str(args.per_order_fee), str(args.commission_rate),
        str(start or ""), str(end or ""),
    ])
    _sig = _hashlib.md5(_key.encode("utf-8")).hexdigest()[:10]
    ckpt = str(out / f"_checkpoint_{stem}_{_sig}.pkl")   # auto-removed on clean finish

    print(f"running {strategy}  window {start or 'ALL'} -> {end or 'ALL'}")
    df = eng.run(strategy, progress_every=20000, checkpoint_path=ckpt)
    df.to_csv(out / f"equity_{stem}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eng.account.blotter,
                 columns=["timetag", "side", "code", "shares", "price", "fee", "tag"]
                 ).to_csv(out / f"blotter_{stem}.csv", index=False, encoding="utf-8-sig")
    rj = pd.DataFrame(eng.account.rejects, columns=["timetag", "code", "reason"])
    rj.to_csv(out / f"rejects_{stem}.csv", index=False, encoding="utf-8-sig")

    # 后复权 TWRR cross-check (only for the raw run; the hfq run's NAV is already 后复权)
    hfq = pd.DataFrame()
    if args.adjust != "hfq":
        codes = sorted({c for _, _, snap in eng.daily_holdings for c in snap})
        raw_close = {c: eng.feed.daily_close_series(c) for c in codes}
        hfq = hfq_twrr_curve(eng.daily_holdings, raw_close,
                             load_adj_factors(TS_DB, codes), args.capital)
        if not hfq.empty:
            hfq.to_csv(out / f"equity_hfq_{stem}.csv", index=False, encoding="utf-8-sig")

    print("\n===== summary =====")
    if not df.empty:
        label = ("后复权价执行 (独立交叉验证)" if args.adjust == "hfq"
                 else "raw真钱执行 + 除权除息 (权威口径)")
        print_metrics(df["nav"], args.capital, label)
    if not hfq.empty:
        print_metrics(hfq["nav_hfq"], args.capital, "后复权 TWRR (交叉验证)")
    print(f"trades filled  : {len(eng.account.blotter)}")
    print(f"order rejects  : {len(rj)}")
    if len(rj):
        print(f"reject reasons : {rj['reason'].value_counts().head(6).to_dict()}")
    print(f"\nequity -> {out / f'equity_{stem}.csv'}")


if __name__ == "__main__":
    main()
