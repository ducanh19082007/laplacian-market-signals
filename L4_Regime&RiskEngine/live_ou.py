"""
Live driver: stream the real venues, compute the tropical eigenvalue every interval
(L2), feed it to the rolling OU predictor (L4), and alert on predicted arbitrage.

This is the LIVE counterpart to OUArbitrage.py's offline/synthetic __main__. It
reuses Engine.default_config() for the exact same broker wiring the main engine
uses, so it streams the public Binance / Coinbase / Kraken books (no API keys).

Run it (from the repo root):
    python "L4_Regime&RiskEngine/live_ou.py"
    python "L4_Regime&RiskEngine/live_ou.py" --interval 1.0 --window 60 --fee 0.002
    python "L4_Regime&RiskEngine/live_ou.py" --seconds 120 --save   # auto-stop + render

Ctrl-C (or --seconds) stops it; with --save it dumps a rolling GIF + summary PNG of
the live tape using the same renderers as the backtest.

Author: Anh Duc Le
"""

import argparse
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from L1_DataProcessing.MultiVenueFeed import MultiBrokerOrderBook, build_default_feed
from L1_DataProcessing.DataProcessing import ExchangeRateGraph
from L2_MarketStructureAnalysis.TropicalEigenvalue import TropicalEigenvalue
from L2_MarketStructureAnalysis.GraphLaplacian import Laplacian

# OUArbitrage lives in a folder whose name ('L4_Regime&RiskEngine') can't be
# imported as a package (the '&'), so load it by path from this same directory.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "ou_arbitrage", os.path.join(os.path.dirname(os.path.abspath(__file__)), "OUArbitrage.py")
)
_ou = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ou)
RollingArbitragePredictor = _ou.RollingArbitragePredictor
backtest = _ou.backtest
render_summary_png = _ou.render_summary_png
render_rolling_gif = _ou.render_rolling_gif

RED = "\033[91m"
GRN = "\033[92m"
DIM = "\033[2m"
RST = "\033[0m"


def fiedler_of(g) -> tuple:
    """L2 Fiedler value lambda_2 (connectivity) + MarketStrain off a live graph -- the
    regime/risk signal. (nan, nan) on a degenerate graph. 'average' = smooth lambda_2(t)."""
    try:
        lap = Laplacian(g, attr="weight", symmetrize="average")
        f = lap.FiedlerValue()
        return (f, lap.MarketStrain) if np.isfinite(f) else (np.nan, np.nan)
    except Exception:
        return (np.nan, np.nan)


def build_feed(feed_fee: float, quote_window: float, max_quote_age: float,
               no_depth_filter: bool) -> MultiBrokerOrderBook:
    """The rich 6-venue example universe from MultiVenueFeed (starts streaming on construct)."""
    kw = dict(fee=feed_fee, quote_window=quote_window, max_quote_age=max_quote_age)
    if no_depth_filter:
        kw["min_notional"] = None       # disable depth filter -> more (theoretical/phantom) arb
    return build_default_feed(**kw)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Live tropical-eigenvalue OU arbitrage monitor. Defaults to a REALISTIC "
                    "regime (anti-phantom guards on); use --loose for the theoretical-arb demo.")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between eigenvalue samples")
    ap.add_argument("--window", type=int, default=60, help="rolling-window length (samples)")
    ap.add_argument("--min-fit", type=int, default=15, help="samples before OU calibrates")
    ap.add_argument("--fee", type=float, default=0.0001,
                    help="arb threshold: per-hop execution fee (tau) that a loop must clear")
    ap.add_argument("--feed-fee", type=float, default=0.00015,
                    help="per-leg taker fee baked into graph edges (0 = frictionless)")
    ap.add_argument("--quote-window", type=float, default=0.2,
                    help="max seconds apart cycle legs may be")
    ap.add_argument("--max-quote-age", type=float, default=1.0, help="drop quotes older than this")
    ap.add_argument("--no-depth-filter", action="store_true",
                    help="disable the min-notional depth filter (lets thin-book phantom arb through)")
    ap.add_argument("--loose", action="store_true",
                    help="low-friction DEMO regime: frictionless edges, wide 5s windows, depth "
                         "filter OFF -- surfaces theoretical/phantom arb that is NOT tradeable")
    ap.add_argument("--strict", action="store_true",
                    help="tighter realistic regime: fee=0.2%%, feed-fee=0.03%%, tight windows, depth on")
    ap.add_argument("--seconds", type=float, default=0.0, help="auto-stop after N seconds (0 = until Ctrl-C)")
    ap.add_argument("--save", action="store_true", help="render GIF+PNG of the live tape on exit")
    args = ap.parse_args()

    # Default = REALISTIC (guards on). --loose re-opens all guards (phantom-arb demo);
    # --strict tightens fee/windows further and wins if both are passed.
    if args.loose:
        args.feed_fee = 0.0
        args.quote_window, args.max_quote_age = 5.0, 5.0
        args.no_depth_filter = True
    if args.strict:
        args.fee, args.feed_fee = 0.002, 0.0003
        args.quote_window, args.max_quote_age = 0.2, 1.0
        args.no_depth_filter = False

    feed = build_feed(args.feed_fee, args.quote_window, args.max_quote_age, args.no_depth_filter)
    predictor = RollingArbitragePredictor(
        window=args.window, dt=args.interval, fee=args.fee, min_fit=args.min_fit
    )
    times, lams = [], []

    print(f"{DIM}warming up the feed... (need {args.min_fit} samples @ {args.interval}s "
          f"before OU calibrates){RST}")
    t0 = time.time()
    try:
        k = 0
        while True:
            now = time.time()
            g = feed.build_graph()
            if g is None:
                time.sleep(args.interval)
                continue

            trop = TropicalEigenvalue(g)
            res = trop.compute()
            lam = res.eigenvalue
            if not np.isfinite(lam):
                time.sleep(args.interval)
                continue

            fiedler, strain = fiedler_of(g)     # regime/risk: connectivity lambda_2

            t = now - t0
            times.append(t)
            lams.append(lam)
            pred = predictor.push(t, lam)

            loop = ""
            if res.has_cycle:
                loop = " -> ".join(ExchangeRateGraph.fmt(n) for n in res.cycle)

            if pred is None:
                print(f"[{time.strftime('%H:%M:%S')}] lambda={lam:+.6f}  "
                      f"lambda2={fiedler:.4f}  strain={strain:.1f}  "
                      f"(warming {len(lams)}/{args.min_fit})")
            else:
                p = pred.params
                hl = p.half_life
                hl_s = f"{hl:5.1f}s" if np.isfinite(hl) else "  inf"
                tag = f"{RED}x ARB PREDICTED{RST}" if pred.predicted_arb else (
                      f"{GRN}ok{RST}" if lam <= pred.threshold else f"{RED}(real arb now){RST}")
                print(f"[{time.strftime('%H:%M:%S')}] lambda={lam:+.6f}  "
                      f"pred_next={pred.mean_next:+.6f}  tau={pred.threshold:+.6f}  "
                      f"theta={p.theta:5.2f}  mu={p.mu:+.5f}  hl={hl_s}  "
                      f"lambda2={fiedler:.4f}  strain={strain:.1f}  {tag}")
                if pred.predicted_arb and loop:
                    print(f"           {RED}top loop:{RST} {loop}  "
                          f"({(res.return_multiple - 1) * 100:+.4f}%)")

            k += 1
            if args.seconds and (time.time() - t0) >= args.seconds:
                break
            sleep_left = args.interval - (time.time() - now)
            if sleep_left > 0:
                time.sleep(sleep_left)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        for _, dashboard in feed.dashboards:
            dashboard.is_running = False

    if args.save and len(lams) > args.min_fit:
        t_arr, lam_arr = np.asarray(times), np.asarray(lams)
        res = backtest(t_arr, lam_arr, fee=args.fee, window=args.window,
                       min_fit=args.min_fit, dt=args.interval)
        _ou.print_backtest_report(res)
        png = render_summary_png(res, path=os.path.join(_ROOT, "ou_live_backtest.png"))
        gif = render_rolling_gif(t_arr, lam_arr, fee=args.fee, window=args.window,
                                 min_fit=args.min_fit, dt=args.interval,
                                 path=os.path.join(_ROOT, "ou_live_rolling.gif"))
        print(f"summary PNG : {png}")
        print(f"rolling GIF : {gif}")


if __name__ == "__main__":
    main()
