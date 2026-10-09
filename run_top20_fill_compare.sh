#!/usr/bin/env bash
# top20, 500k, 2018-01..2026-09: original buy sizing vs the fill reallocation.
# Same plan, same cost model (no regulatory fee), run side by side.
set -u
cd "$(dirname "$0")"
P="C:/AI_STOCK/machine_learning_stock_selection/12models_styleneutral_vol+illiq/automatically_plan_generate/backtest_plans"
run() {   # strategy tag
  COMBO_PLAN_CSV="$P/qmt_plan_top20.csv" RUN_TAG="$2" PYTHONIOENCODING=utf-8 \
    python run_ml_backtest.py --strategy "$1" --capital 500000 \
      --start 20180101 --end 20260930 > "data/log_$2.txt" 2>&1
  echo "--- $2 done $(date +%H:%M:%S)"
  sed -n '/===== summary =====/,$p' "data/log_$2.txt" | head -8
}
echo "start $(date +%H:%M:%S)"
run qmt_combo_top20_twap.py      12m_top20_base &
run qmt_combo_top20_twap_fill.py 12m_top20_fill &
wait
echo "ALL DONE $(date +%H:%M:%S)"
