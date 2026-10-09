#coding:utf-8
"""
Replay ONE day of the backtest TWAP SELL against a real liquidation.

Loads qmt_combo_top20_twap.py unchanged and runs only its sell side
(_run_sells), exactly as handlebar() would on a signal day: every name held at
the open goes to zero over SELL_START..SELL_RUSH_END. The account is seeded by
the caller (tools/replay_sell_day.py) with the live holdings, so the backtest
sells the same names and the same share counts the live script did, and the
two can be compared fill for fill.

REPLAY_DAY (env) is the trading day to replay.
"""
import os as _os_r

_SRC = _os_r.path.join(_os_r.path.dirname(_os_r.path.abspath(__file__)) if "__file__" in globals()
                       else r"C:\AI_STOCK\1min_qmt_backtest_engine\strategies",
                       "qmt_combo_top20_twap.py")
exec(compile(open(_SRC, "rb").read(), _SRC, "exec"), globals())
REPLAY_DAY = _os_r.environ["REPLAY_DAY"]


def init(C):
    C.account_id = ACCOUNT_ID
    C.pending_sells = {}
    C.sell_cur0 = {}
    C.sell_cur0_date = None
    C.sell_mark = {}
    C.delayed_sell_log = []
    C.suspend_log = []
    C._data_today = {}
    C._replay_planned = False


def handlebar(C):
    today, hhmmss = _bar_datetime(C)
    if today != REPLAY_DAY:
        return
    if not C._replay_planned and hhmmss >= SELL_START:
        # what _plan_sells does on a signal day: every held name -> 0
        held = _positions(C)
        C.pending_sells = dict((c, 0) for c in held)
        C.sell_cur0 = dict(held)
        C.sell_cur0_date = today
        C._replay_planned = True
        print("replay: selling", len(held), "names", sum(held.values()), "shares")
    if SELL_START <= hhmmss <= SELL_RUSH_END:
        _run_sells(C, today, hhmmss)


def stop(C):
    left = _positions(C)
    print("replay done: unsold", left if left else "none")
