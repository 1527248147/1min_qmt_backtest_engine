#coding:utf-8
"""
Corporate-action data loaders (送转/分红 + 复权因子) from a Tushare DuckDB.

Kept here (not in the run script) so any strategy/run in this project can reuse
them.  db_path is a parameter -> the engine stays data-source agnostic.

* load_corp_actions(db): per-ex-date 送转比例 + 税前现金分红, fed to
  account.apply_corporate_actions so the raw-priced NAV tracks total return.
* load_adj_factors(db): daily 后复权因子, used by the hfq cross-check and by the
  ParquetFeed `adjust="hfq"` mode.
"""

from __future__ import annotations

import duckdb


def load_corp_actions(db_path):
    """{code: {ex_date_int: (stk_div, cash_div_pretax)}} -- 实施 dividends only."""
    con = duckdb.connect()
    con.execute(f"ATTACH '{db_path}' AS ts (READ_ONLY)")
    df = con.execute(
        "SELECT ts_code, ex_date, COALESCE(stk_div,0) AS sd, "
        "COALESCE(cash_div_tax,0) AS cd FROM ts.value.dividend_combined "
        "WHERE div_proc = '实施' AND ex_date IS NOT NULL AND ex_date <> ''"
    ).df()
    con.close()
    corp = {}
    for r in df.itertuples(index=False):
        try:
            ex = int(r.ex_date)
        except (TypeError, ValueError):
            continue
        d = corp.setdefault(r.ts_code, {})
        s, c = d.get(ex, (0.0, 0.0))
        d[ex] = (s + float(r.sd), c + float(r.cd))
    return corp


def load_adj_factors(db_path, codes=None):
    """{code: {trade_date_int: adj_factor}}; pass codes to restrict the load."""
    con = duckdb.connect()
    con.execute(f"ATTACH '{db_path}' AS ts (READ_ONLY)")
    q = "SELECT ts_code, trade_date, adj_factor FROM ts.market.adj_factor"
    if codes:
        q += " WHERE ts_code IN (" + ",".join("'" + c + "'" for c in codes) + ")"
    df = con.execute(q).df()
    con.close()
    out = {}
    for c, g in df.groupby("ts_code"):
        out[c] = {int(d): float(f) for d, f in zip(g.trade_date, g.adj_factor)}
    return out
