#coding:utf-8
"""
Performance analytics for the backtest engine -- reusable across strategies.

* metrics / print_metrics: total return, annualised, Sharpe, max-drawdown of a
  NAV series.
* hfq_twrr_curve: a 后复权 time-weighted-return cross-check of the engine's raw
  NAV.  "TWRR" (time-weighted rate of return) just chains each day's portfolio
  percentage return.  Because the engine already credits 送转/分红 to the raw
  positions (account.apply_corporate_actions), its raw NAV *is* the 后复权 total
  return in real yuan; this curve is only an independent re-derivation to
  validate it.  On non-rebalance days it equals the raw daily return exactly;
  small differences on rebalance days are the TWRR lagged-holdings approximation.
"""

from __future__ import annotations

import pandas as pd


def metrics(nav_series, initial_capital, periods_per_year=252):
    nav = pd.Series(list(nav_series)).reset_index(drop=True)
    if len(nav) == 0:
        return {}
    nav1 = float(nav.iloc[-1])
    days = len(nav)
    ret = nav.pct_change().fillna(0.0)
    total = nav1 / initial_capital - 1
    ann = (nav1 / initial_capital) ** (periods_per_year / max(days, 1)) - 1
    vol = ret.std() * (periods_per_year ** 0.5)
    sharpe = ann / vol if vol else 0.0
    mdd = ((nav - nav.cummax()) / nav.cummax()).min()
    return {"days": days, "final": nav1, "total": total, "ann": ann,
            "vol": float(vol), "sharpe": float(sharpe), "maxdd": float(mdd)}


def print_metrics(nav_series, initial_capital, label, periods_per_year=252):
    m = metrics(nav_series, initial_capital, periods_per_year)
    if m:
        print(f"[{label}] days {m['days']}  final {m['final']:,.0f}  "
              f"total {m['total']*100:.2f}%  ann {m['ann']*100:.2f}%  "
              f"sharpe {m['sharpe']:.2f}  maxDD {m['maxdd']*100:.2f}%")
    return m


def hfq_twrr_curve(daily_holdings, raw_close, factors, initial_capital):
    """后复权 TWRR cross-check curve.

    daily_holdings: [(date, cash, {code: (shares, raw_close)})]  (engine.daily_holdings)
    raw_close:      {code: {date_int: close}}                    (feed daily closes)
    factors:        {code: {date_int: adj_factor}}               (corporate_actions.load_adj_factors)
    returns:        DataFrame[date, nav_hfq]
    """
    dh = daily_holdings
    if len(dh) < 2:
        return pd.DataFrame(columns=["date", "nav_hfq"])

    def factor(code, day):
        s = factors.get(code)
        if not s:
            return 1.0
        if day in s:
            return s[day]
        prev = [d for d in s if d <= day]
        return s[max(prev)] if prev else 1.0

    def hfq(code, day):
        rc = raw_close.get(code, {}).get(day)
        return rc * factor(code, day) if rc is not None else None

    nav = [float(initial_capital)]
    out_dates = [int(dh[0][0])]
    for k in range(1, len(dh)):
        d0, cash0, s0 = dh[k - 1]
        d1 = dh[k][0]
        d0i, d1i = int(d0), int(d1)
        nav0 = cash0 + sum(sh * pc for sh, pc in s0.values())
        port_ret = 0.0
        if nav0 > 0:
            for code, (sh, pc0) in s0.items():
                h0, h1 = hfq(code, d0i), hfq(code, d1i)
                if h0 and h1 and h0 > 0:
                    port_ret += (sh * pc0 / nav0) * (h1 / h0 - 1.0)
        nav.append(nav[-1] * (1.0 + port_ret))
        out_dates.append(d1i)
    return pd.DataFrame({"date": out_dates, "nav_hfq": nav})
