#!/usr/bin/env bash
# 12models_styleneutral_vol+illiq signals, TWAP, three larger baskets (restart after the feed fix).
# Minute data from DuckDB (DuckDBFeed); outputs tagged 12m_<basket>.
set -u
cd "$(dirname "$0")"
P="C:/AI_STOCK/machine_learning_stock_selection/12models_styleneutral_vol+illiq/automatically_plan_generate/backtest_plans"
START=20180101
END=20260930
run() {   # tag capital
  COMBO_PLAN_CSV="$P/qmt_plan_$1.csv" RUN_TAG="12m_$1" PYTHONIOENCODING=utf-8 \
    python run_ml_backtest.py --strategy qmt_combo_top20_twap.py \
      --capital "$2" --start $START --end $END > "data/log_12m_$1.txt" 2>&1
  echo "--- $1 done $(date +%H:%M:%S)"
  sed -n '/===== summary =====/,$p' "data/log_12m_$1.txt" | head -8
}
echo "start $(date +%H:%M:%S)"
run pct10  50000000 &
run pct5   20000000 &
run pct2.5 10000000 &
wait
echo "ALL DONE $(date +%H:%M:%S)"
