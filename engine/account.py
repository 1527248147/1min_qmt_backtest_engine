#coding:utf-8
"""
Account + matcher.

This is the one piece QMT keeps in its closed C++ engine -- so we reimplement
A-share matching here: T+1 settlement, 100-share lot rounding, commission, sell
stamp tax, and limit-up/limit-down lock checks.

An order placed during ``handlebar`` at bar T is matched immediately against the
bar of the order's own code at the same timetag T (this is how QMT model
backtests behave for 对手价/市价 style orders).  If that code has no bar at T,
the order does not fill and is reported back so the strategy can retry on a
later bar -- exactly what the ML strategy's ``pending_sells`` logic expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .datafeed import timetag_to_beijing

TICK = 0.01   # A-share price tick (stocks/funds)


def board_limit_pct(code: str, is_st: bool = False) -> float:
    """Daily price-limit percent for an A-share, board-aware.

    主板 ±10%  |  创业板 300xxx.SZ / 科创板 688xxx.SH ±20%  |
    北交所 .BJ ±30%  |  ST/*ST ±5% (overrides board).
    """
    if is_st:
        return 0.05
    stk, market = code.split(".")
    market = market.upper()
    if market == "BJ":
        return 0.30
    if market == "SH" and stk.startswith("688"):
        return 0.20
    # 301xxx is ChiNext too. Testing only "300" priced every 301 name at the
    # main-board +-10%, a ceiling 10% too low, which reads as "limit-up, cannot
    # buy" on any strong day. Same omission existed in the strategy's own copy.
    if market == "SZ" and (stk.startswith("300") or stk.startswith("301")):
        return 0.20
    return 0.10


def limit_prices(code: str, preclose: float, is_st: bool = False):
    """Return (up_limit, down_limit) rounded to the A-share tick."""
    pct = board_limit_pct(code, is_st)
    up = round(preclose * (1 + pct) / TICK) * TICK
    down = round(preclose * (1 - pct) / TICK) * TICK
    return round(up, 2), round(down, 2)


def load_st_ranges(path=None):
    """code -> ((start_yyyymmdd, end_yyyymmdd), ...) from data/st_ranges.csv.

    Built from Tushare's namechange table (meta.namechange in tushare.duckdb):
    every row whose name begins with ST or *ST, with a missing end_date meaning
    "still ST", stored as 99999999. Missing file -> empty dict, and every stock
    is then priced at its board limit, which is what the engine did before.
    """
    import csv as _csv
    import os as _os
    if path is None:
        path = _os.path.join(_os.path.dirname(_os.path.dirname(
            _os.path.abspath(__file__))), "data", "st_ranges.csv")
    out = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                try:
                    a = int(row["start_date"]); b = int(row["end_date"])
                except (KeyError, ValueError, TypeError):
                    continue
                out.setdefault(row["ts_code"], []).append((a, b))
    except OSError:
        return {}
    return dict((k, tuple(v)) for k, v in out.items())


def buy_unit(code: str):
    """Board-aware (min_shares, step) for a BUY order.

    沪/深主板·创业板: 最低100股, 必须100整数倍 -> (100, 100)
    科创板 688.SH:    最低200股, 超200后按1股递增 -> (200, 1)
    北交所 .BJ:       最低100股, 按1股递增        -> (100, 1)
    """
    stk, market = code.split(".")
    market = market.upper()
    if market == "SH" and stk.startswith("688"):
        return (200, 1)
    if market == "BJ":
        return (100, 1)
    return (100, 100)


def round_buy(code: str, shares) -> int:
    """Round a desired buy quantity DOWN to a board-valid order size (0 if < min)."""
    mn, step = buy_unit(code)
    shares = int(shares)
    if shares < mn:
        return 0
    return mn + ((shares - mn) // step) * step


def round_sell(code: str, shares, clearing: bool) -> int:
    """Round a SELL quantity down to a legal order size (0 if nothing is legal).

    主板/创业板   拆单必须是 100 的整数倍，且不得低于 100
    科创板         步长 1 股，但【拆单不得低于 200 股】
    北交所         步长 1 股，但【拆单不得低于 100 股】
    清仓(clearing) 允许带零头，任意数量

    最后那个下限以前漏了：step==1 的板块直接 `return shares`，于是 75 股的科创板
    拆单在回测里可以成交。现实中不行 —— 科创板规定余额不足 200 股的部分必须一次性
    卖出，也就是只有清仓才能带零头。实盘脚本 combo_sell_dual_model._round_sell 一直
    是按这个下限做的（它的最后一行就是 `qty if qty >= _min_lot else 0`），回测比实盘
    宽松，方向上高估了科创板的可卖出程度。
    """
    shares = int(shares)
    if clearing:
        return shares
    mn, step = buy_unit(code)
    shares = (shares // step) * step
    return shares if shares >= mn else 0


@dataclass
class CostModel:
    commission_rate: float = 0.00008  # 双边佣金 万0.8 = 0.8/10000
    min_commission: float = 5.0       # 单笔最低佣金 5元
    transfer_fee: float = 0.00001     # 过户费 万0.1 (双边)
    # 规费 万0.625 = 0.625/10000, 双边. 交易所经手费 + 证监会证管费, 券商代收,
    # 按成交金额计, 与佣金分开列示且不参与最低佣金. 用户实际费率, 2026-09-02 加入.
    # 双边合计 1.25bp -- 和佣金万0.8 双边的 1.6bp 同一量级, 不是可以忽略的零头.
    regulatory_fee: float = 0.0000625
    slippage_bps: float = 0.0         # 滑点(基点),买入加价/卖出减价
    # 每笔委托的固定费用(元). 券商流量费/申报费之类, 与成交金额无关.
    # 该用户: 无最低佣金, 但每笔 0.1 元流量费.
    per_order_fee: float = 0.0
    # 卖出印花税: 现实规定, 2023-08-28 起由 0.001 减半为 0.0005 (单边卖出)
    stamp_tax_old: float = 0.0010
    stamp_tax_new: float = 0.0005
    stamp_change_date: int = 20230828

    def stamp_rate(self, trade_date: int) -> float:
        return self.stamp_tax_new if int(trade_date) >= self.stamp_change_date else self.stamp_tax_old

    def buy_cost(self, amount: float) -> float:
        return (max(amount * self.commission_rate, self.min_commission)
                + amount * self.transfer_fee
                + amount * self.regulatory_fee + self.per_order_fee)

    def sell_cost(self, amount: float, trade_date: int) -> float:
        return (max(amount * self.commission_rate, self.min_commission)
                + amount * self.stamp_rate(trade_date) + amount * self.transfer_fee
                + amount * self.regulatory_fee + self.per_order_fee)


@dataclass
class Position:
    code: str
    volume: int = 0            # 总持仓
    can_use: int = 0           # 可卖(T+1,昨仓)
    cost: float = 0.0          # 持仓均价
    last_price: float = 0.0    # 最新价(盯市)

    @property
    def market_value(self) -> float:
        return self.volume * self.last_price


class DetailData:
    """Mimics QMT's trade-detail object so strategies read m_* attributes."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


@dataclass
class Account:
    cash: float
    cost: CostModel = field(default_factory=CostModel)
    positions: dict = field(default_factory=dict)   # code -> Position
    blotter: list = field(default_factory=list)      # filled trades log
    rejects: list = field(default_factory=list)      # (timetag, code, reason)
    st_codes: set = field(default_factory=set)        # codes ALWAYS treated as ST
    # code -> ((start_yyyymmdd, end_yyyymmdd), ...), from data/st_ranges.csv.
    # ST status is not a property of a stock, it is a property of a stock ON A
    # DATE: 600530.SH was *ST from 2020-04-30 to 2021-05-16 and normal on either
    # side. Keying it statically would price twenty years of a name at +-5%
    # because of one bad year. Loaded by load_st_ranges() below.
    st_ranges: dict = field(default_factory=dict)
    max_vol_rate: float = 1.0     # QMT 默认: 单笔最多成交 = 当bar成交量 × 该比例
    _st_memo: dict = field(default_factory=dict)      # (code, date) -> bool

    # ----- limit / tradability checks ----------------------------------

    def _bar_date(self, bar):
        """yyyymmdd of a bar, as an int. Bars arrive in time order, so a
        one-entry memo on the timetag hits on essentially every call."""
        tt = bar.timetag
        if tt == getattr(self, "_last_tt", None):
            return self._last_date
        d = timetag_to_beijing(tt)
        self._last_tt = tt
        self._last_date = d.year * 10000 + d.month * 100 + d.day
        return self._last_date

    def _is_st(self, code, bar):
        if code in self.st_codes:
            return True
        rng = self.st_ranges.get(code)
        if not rng:                      # ~99% of calls stop here
            return False
        d = self._bar_date(bar)
        key = (code, d)
        hit = self._st_memo.get(key)
        if hit is None:
            hit = any(a <= d <= b for a, b in rng)
            self._st_memo[key] = hit
        return hit

    def _limits(self, code, bar):
        return limit_prices(code, bar.preclose, self._is_st(code, bar))

    def can_buy(self, code, bar):
        """Why a 对手价 buy would fail this bar, or '' if fillable."""
        if bar is None:
            return "no_bar"
        if bar.volume == 0:
            return "no_volume"          # 停牌/无成交,无对手
        up, _ = self._limits(code, bar)
        # 涨停买不进。判据是 CLOSE,不是 LOW。
        #
        # 成交价取的是 bar.close(见 _fill_price),所以"收在涨停价"就等于"在封单
        # 上买到货" —— 板上排着几万手,轮不到我们。原来的 low >= up 要求整分钟都在
        # 板上,中间只要有一笔打在板下,整个切片就按收盘价(=涨停价)全额成交了。
        # 换成 close 只会更保守: 分钟收在板上 -> 买不到; 开过板并回落 -> 按回落后
        # 的收盘价成交,那是真能买到的。
        if bar.close >= up - TICK / 2:
            return "limit_up_lock"
        return ""

    def can_sell(self, code, bar):
        """Why a 对手价/跌停价 sell would fail this bar, or '' if fillable."""
        if bar is None:
            return "no_bar"
        if bar.volume == 0:
            return "no_volume"
        _, down = self._limits(code, bar)
        # 跌停卖不出。判据同样改成 CLOSE,和 can_buy 对称。
        # 只把买入侧改严、卖出侧留松,回测就变成"买难卖易",会系统性高估收益。
        if bar.close <= down + TICK / 2:
            return "limit_down_lock"
        return ""

    # ----- daily T+1 rollover ------------------------------------------

    def settle_new_day(self):
        """Make yesterday's buys sellable; called at each new trading day."""
        for p in self.positions.values():
            p.can_use = p.volume

    def apply_corporate_actions(self, day, corp_actions):
        """On an ex-date, adjust held positions for 送转/分红 so the (raw-priced)
        NAV stays continuous through the price gap:
            送转: shares *= (1 + stk_div); cost/share /= (1 + stk_div)
            分红: cash  += shares * cash_div (pre-tax, matches the price drop)
        corp_actions: {code: {ex_date_int: (stk_div, cash_div)}}
        """
        for code, p in list(self.positions.items()):
            ca = corp_actions.get(code, {}).get(day)
            if not ca:
                continue
            stk_div, cash_div = ca
            if cash_div > 0:
                self.cash += p.volume * cash_div
            if stk_div and stk_div > 0:
                f = 1.0 + stk_div
                p.volume = int(round(p.volume * f))
                p.can_use = int(round(p.can_use * f))
                p.cost = p.cost / f
            self.blotter.append((day, "corp_action", code, p.volume, cash_div, stk_div, "ex"))

    # ----- mark to market ----------------------------------------------

    def mark(self, code: str, price: float):
        p = self.positions.get(code)
        if p and price > 0:
            p.last_price = price

    def total_value(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    def stock_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    # ----- matching ----------------------------------------------------

    def _fill_price(self, code: str, side: str, bar) -> float:
        slip = self.cost.slippage_bps / 1e4
        ref = bar.close
        px = ref * (1 + slip) if side == "buy" else ref * (1 - slip)
        # Clamp into the legal band. Without this, a buy on a bar that closed
        # near the ceiling fills ABOVE the ceiling once slippage is added, and a
        # sell near the floor fills BELOW it -- prices that cannot exist. Small
        # in size, but it removes a way for the model to book a fill the
        # exchange would never have allowed.
        up, down = self._limits(code, bar)
        if px > up:
            px = up
        elif px < down:
            px = down
        return px

    def buy(self, code, target_shares, bar, timetag, tag=""):
        """target_shares already lot-rounded delta (>0). bar = code's bar at timetag."""
        reason = self.can_buy(code, bar)
        if reason:
            self.rejects.append((timetag, code, reason))
            return 0
        price = self._fill_price(code, "buy", bar)
        shares = round_buy(code, target_shares)
        if shares <= 0:
            return 0
        # QMT max_vol_rate: cannot take more than this fraction of the bar's volume
        shares = round_buy(code, min(shares, self.max_vol_rate * bar.volume))
        if shares <= 0:
            self.rejects.append((timetag, code, "volume_cap"))
            return 0
        amount = shares * price
        fee = self.cost.buy_cost(amount)
        if amount + fee > self.cash + 1e-6:
            # shrink to an affordable, board-valid size
            shares = round_buy(code, self.cash / (price * 1.002))
            if shares <= 0:
                self.rejects.append((timetag, code, "insufficient_cash"))
                return 0
            amount = shares * price
            fee = self.cost.buy_cost(amount)
        self.cash -= amount + fee
        p = self.positions.get(code) or Position(code)
        new_vol = p.volume + shares
        p.cost = (p.cost * p.volume + amount) / new_vol if new_vol else 0.0
        p.volume = new_vol
        p.last_price = price
        # bought today -> not sellable today (T+1); can_use unchanged
        self.positions[code] = p
        self.blotter.append((timetag, "buy", code, shares, price, fee, tag))
        return shares

    def sell(self, code, shares, bar, timetag, tag=""):
        """shares>0; only can_use (T+1) is sellable. bar = code's bar at timetag."""
        p = self.positions.get(code)
        if not p or p.can_use <= 0:
            self.rejects.append((timetag, code, "no_available_position"))
            return 0
        reason = self.can_sell(code, bar)
        if reason:
            self.rejects.append((timetag, code, reason))
            return 0
        shares = min(shares, p.can_use)
        # QMT max_vol_rate: cap fill at this fraction of the bar's volume.
        # This is what makes thin closing-auctions only partially fill and a
        # volume==0 auction unsellable (-> stays pending, retried next day).
        shares = min(shares, int(self.max_vol_rate * bar.volume))
        # board-aware sell rounding; clearing the whole available position may
        # sell an odd tail, otherwise 主板/创业板 must be 100s
        clearing = shares >= p.can_use
        shares = round_sell(code, shares, clearing)
        if shares <= 0:
            self.rejects.append((timetag, code, "volume_cap"))
            return 0
        price = self._fill_price(code, "sell", bar)
        amount = shares * price
        trade_date = int(timetag_to_beijing(timetag).strftime("%Y%m%d"))
        fee = self.cost.sell_cost(amount, trade_date)
        self.cash += amount - fee
        p.volume -= shares
        p.can_use -= shares
        p.last_price = price
        if p.volume <= 0:
            del self.positions[code]
        self.blotter.append((timetag, "sell", code, shares, price, fee, tag))
        return shares

    # ----- QMT-style detail views --------------------------------------

    def detail_account(self):
        return [DetailData(
            m_dBalance=self.total_value(),       # 总资产
            m_dAvailable=self.cash,              # 可用资金
            m_dStockValue=self.stock_value(),    # 持仓市值
            m_dInstrumentValue=self.stock_value(),
        )]

    def detail_positions(self):
        out = []
        for code, p in self.positions.items():
            stk, market = code.split(".")
            out.append(DetailData(
                m_strInstrumentID=stk,
                m_strExchangeID=market,
                m_nVolume=int(p.volume),
                m_nCanUseVolume=int(p.can_use),
                m_dOpenPrice=p.cost,
                m_dMarketValue=p.market_value,
                m_dLastPrice=p.last_price,
            ))
        return out
