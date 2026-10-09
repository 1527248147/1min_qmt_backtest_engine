#!/usr/bin/env bash
# top20, 500k, 2018-01..2026-10-08: minimum slice 100 vs 200 shares, with and
# without "tail goes out whole", the SAME rule on the buy and the sell side.
set -u
cd "$(dirname "$0")"
P="C:/AI_STOCK/machine_learning_stock_selection/12models_styleneutral_vol+illiq/automatically_plan_generate/backtest_plans"
run() {   # tag min tail
  COMBO_PLAN_CSV="$P/qmt_plan_top20.csv" RUN_TAG="$1" PYTHONIOENCODING=utf-8 \
  SELL_MIN_SLICE=$2 BUY_MIN_SLICE=$2 SELL_TAIL_WHOLE=$3 BUY_TAIL_WHOLE=$3 \
    python run_ml_backtest.py --strategy qmt_combo_top20_twap.py --capital 500000 \
      --start 20180101 --end 20261008 > "data/log_$1.txt" 2>&1
  echo "--- $1 done $(date +%H:%M:%S)"
  sed -n '/===== summary =====/,$p' "data/log_$1.txt" | head -7
}
echo "start $(date +%H:%M:%S)"
run ms_min100_tail0 100 0 &
run ms_min100_tail1 100 1 &
run ms_min200_tail0 200 0 &
run ms_min200_tail1 200 1 &
wait
echo "ALL DONE $(date +%H:%M:%S)"
