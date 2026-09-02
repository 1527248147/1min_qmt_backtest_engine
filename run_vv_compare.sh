#!/usr/bin/env bash
# 新(均价标签) vs 旧(变体C, open/close标签) 各两档, 同一引擎同一窗口
set -u
G="C:/AI_STOCK/machine_learning_stock_selection/18models_avgprice_label_alpha192+fund310+forecast/automatically_plan_generate"
END=20260602
for TAG in vv_pct2.5 vv_pct5 oldvarC_pct2.5 oldvarC_pct5; do
  echo "##################### $TAG  $(date +%H:%M:%S)"
  COMBO_PLAN_CSV="$G/qmt_plan_$TAG.csv" PYTHONIOENCODING=utf-8 \
    python run_ml_backtest.py --strategy qmt_combo_top20_twap.py \
      --capital 20000000 --start 20180101 --end $END > "data/log_$TAG.txt" 2>&1
  for f in equity hfq_equity equity_hfq blotter rejects; do
    [ -f "data/${f}_qmt_combo_top20_twap.csv" ] && mv "data/${f}_qmt_combo_top20_twap.csv" "data/${f}_$TAG.csv"
  done
  echo "--- $TAG summary ---"
  sed -n '/===== summary =====/,$p' "data/log_$TAG.txt" | head -8
done
echo "ALL DONE $(date +%H:%M:%S)"
