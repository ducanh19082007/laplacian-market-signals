"""
Dual-view live dashboard: the L1 MultiVenueFeed arbitrage table in the TERMINAL and
the L4 OU rolling eigenvalue graph in a POP-UP WINDOW, both driven off ONE shared
sampler so they refresh from the SAME tick at the SAME time.

    +--------------------------+        +-------------------------------------+
    |  TERMINAL                |        |  MATPLOTLIB WINDOW  (main thread)   |
    |  live exchange-rate box  |   <--  |  top: lambda(t) + OU forecast fan   |
    |  + arbitrage detections  |  same  |       + 95% band + red-x + green o  |
    |  + live OU status (l,th) |  tick  |  bottom: cumulative captured edge   |
    +--------------------------+        +-------------------------------------+
             ^                                        ^
             |         ONE sampler thread             |
             +--- build_graph -> eigenvalue -> OU ----+
                   (single source of truth: LiveState)

Why one sampler: previously the table looped at the feed's 0.01s refresh while the
graph resampled on its own 1s timer, so they drifted and the graph looked laggy /
simpler. Now a single thread samples once per --interval, and BOTH views read that
same LiveState -- so the number on the table is the point on the graph.

The popup is now the SAME rich two-panel figure as the offline ou_backtest.png:
historical 1-step OU forecast, the 95% band, the forward forecast fan, red-x on
predicted arbs, green rings on real arbs, and a cumulative-edge ledger underneath.

Run (from the repo root, needs a display -- WSLg on Win11, or an X server):
    python "L4_Regime&RiskEngine/live_dashboard.py"                # REALISTIC regime (guards on)
    python "L4_Regime&RiskEngine/live_dashboard.py" --loose        # old demo: theoretical/phantom arb
    python "L4_Regime&RiskEngine/live_dashboard.py" --strict       # tighter realistic frictions
    python "L4_Regime&RiskEngine/live_dashboard.py" --headless --seconds 60 --save

Author: Anh Duc Le
"""

import argparse
import os
import sys
import threading
import time
from collections import deque

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from L1_DataProcessing.MultiVenueFeed import build_default_feed
from L1_DataProcessing.DataProcessing import ExchangeRateGraph
from L2_MarketStructureAnalysis.TropicalEigenvalue import TropicalEigenvalue, fee_threshold
from L2_MarketStructureAnalysis.GraphLaplacian import Laplacian

# OUArbitrage.py lives in a folder ('L4_Regime&RiskEngine') that can't be imported as
# a package (the '&'), so load it by path.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "ou_arbitrage", os.path.join(os.path.dirname(os.path.abspath(__file__)), "OUArbitrage.py")
)
_ou = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ou)
OUProcess = _ou.OUProcess
RollingArbitragePredictor = _ou.RollingArbitragePredictor
backtest = _ou.backtest
render_summary_png = _ou.render_summary_png
render_rolling_gif = _ou.render_rolling_gif
_colors = _ou._colors

try:
    from L3_TarjanSCC.TarjanSCC import find_all_arbitrage
except ImportError:                                  # C++ ext not built -> L1 fallback
    def find_all_arbitrage(g):
        c = g.find_arbitrage()
        return [(c, g.cycle_return(c))] if c else []


def fiedler_of(g) -> tuple:
    """
    L2 Fiedler value lambda_2 (algebraic connectivity) + MarketStrain (=1/lambda_2)
    off a live graph. This is the REGIME/RISK signal: a HIGH lambda_2 = a tightly
    connected market (arbitrage keeps prices linked); a DROP = the graph fragmenting
    (venues/assets decoupling -> liquidity stress). Unlike the tropical eigenvalue's
    single-tick arb spikes, lambda_2 is persistent, so it is actually usable.

    Returns (nan, nan) on a degenerate graph (<2 nodes / no valid decomposition).
    'average' symmetrization gives the smooth lambda_2(t) the author recommends.
    """
    try:
        lap = Laplacian(g, attr="weight", symmetrize="average")
        f = lap.FiedlerValue()
        return (f, lap.MarketStrain) if np.isfinite(f) else (np.nan, np.nan)
    except Exception:
        return (np.nan, np.nan)


# ===========================================================================
# Shared state -- the single source of truth both views read
# ===========================================================================
class LiveState:
    def __init__(self) -> None:
        self.times: list = []
        self.lams: list = []
        self.fiedlers: list = []       # L2 Fiedler value lambda_2(t): market connectivity
        self.strains: list = []        # MarketStrain = 1/lambda_2: high => near fragmenting
        self.pred_mean: list = []      # OU 1-step forecast made AT each tick (NaN warmup)
        self.pred_upper: list = []     # its 95% upper band
        self.pred_arb: list = []       # bool: OU predicted next tick clears tau
        self.real_arb: list = []       # bool: this tick a NEW distinct loop hit the table
        # The LEADING forecast: at each tick, OU's call for lambda `lead` seconds LATER.
        # Plotted at x=t+lead so the prediction line runs ahead of the live tape.
        self.lead_mean: list = []
        self.lead_lower: list = []
        self.lead_upper: list = []
        self.det_log: deque = deque(maxlen=15)   # arbitrage detections for the table
        self.status: str = "warming up..."       # live OU status line for the table
        self.stop = threading.Event()
        # MONOTONIC clock, not time.time(): the plotted x-axis is elapsed seconds, and a
        # wall clock can step BACKWARD (WSL2 re-syncs to the Windows host after sleep/idle,
        # NTP corrections). One backward step gives a sample a smaller t than the previous
        # one, so the lambda line draws leftward and crosses itself. monotonic() never
        # decreases, so the tape stays strictly left-to-right.
        self.t0 = time.monotonic()


# ===========================================================================
# The one sampler: build graph -> eigenvalue -> OU -> (optionally) render table
# ===========================================================================
def sampler_loop(feed, state: LiveState, args, predictor, render_table: bool) -> None:
    tau = fee_threshold(args.fee)
    lead_steps = max(1, int(round(args.lead / args.interval)))   # how many ticks = `lead` seconds
    last_sig: set = set()
    while not state.stop.is_set():
        now = time.monotonic()
        g = feed.build_graph()
        if g is None:
            time.sleep(args.interval)
            continue
        res = TropicalEigenvalue(g).compute()
        lam = res.eigenvalue
        if not np.isfinite(lam):
            time.sleep(args.interval)
            continue

        # REGIME/RISK signal: Fiedler value lambda_2 (+ MarketStrain) off the SAME graph.
        # No warmup needed -- it's an instantaneous property of each tick's graph.
        fiedler, strain = fiedler_of(g)

        t = now - state.t0
        p = predictor.push(t, lam)
        # Append the whole per-tick record so both views (and --save) share it.
        state.times.append(t)
        state.lams.append(lam)
        state.fiedlers.append(fiedler)
        state.strains.append(strain)
        if p is None:
            state.pred_mean.append(np.nan)
            state.pred_upper.append(np.nan)
            state.pred_arb.append(False)
            state.lead_mean.append(np.nan)
            state.lead_lower.append(np.nan)
            state.lead_upper.append(np.nan)
            state.status = (f"lambda={lam*100:+.4f}%/hop  lambda2={fiedler:.4f}  "
                            f"strain={strain:.1f}  (warming {len(state.lams)}/{args.min_fit})")
        else:
            state.pred_mean.append(p.mean_next)
            state.pred_upper.append(p.upper_next)
            state.pred_arb.append(p.predicted_arb)
            # The leading call: forecast lead_steps ahead from now, keep the endpoint --
            # OU's expected lambda `lead` seconds into the future (drawn at t+lead).
            lm, ll, lu = OUProcess.forecast(p.params, lam, horizon=lead_steps)
            state.lead_mean.append(float(lm[-1]))
            state.lead_lower.append(float(ll[-1]))
            state.lead_upper.append(float(lu[-1]))
            pr = p.params
            hl = pr.half_life
            hl_s = f"{hl:.1f}s" if np.isfinite(hl) else "inf"
            flag = "  <ARB>" if p.predicted_arb else ""
            state.status = (f"lambda={lam*100:+.4f}%  pred={p.mean_next*100:+.4f}%  "
                            f"tau={tau*100:+.4f}%  theta={pr.theta:.2f}  "
                            f"mu={pr.mu*100:+.4f}%  hl={hl_s}  "
                            f"lambda2={fiedler:.4f}  strain={strain:.1f}{flag}")

        # TRUE-ARBITRAGE detections for the table -- read from the SAME L2 result the
        # graph plots. A loop counts iff its PER-HOP mean (its tropical eigenvalue) clears
        # tau, and we DEDUP by signature: a loop already logged last tick is not re-logged.
        # `state.real_arb` records, per tick, whether a NEW distinct loop hit the table --
        # that flag is exactly what the graph rings in green, so "green ring == a row the
        # table just added", not "every tick the best loop happens to clear tau".
        cur_sig: set = set()
        new_row = False
        for scc in res.per_scc:
            if scc.eigenvalue <= tau or len(scc.cycle) < 2:
                continue
            sig = feed._cycle_signature(scc.cycle)
            cur_sig.add(sig)
            if sig in last_sig:
                continue
            new_row = True
            path = " -> ".join(ExchangeRateGraph.fmt(n) for n in scc.cycle)
            per_hop = scc.eigenvalue * 100.0
            total = (scc.return_multiple - 1.0) * 100.0
            state.det_log.append(f"[{time.strftime('%H:%M:%S')}] "
                                 f"{per_hop:+.4f}%/hop  (tot {total:+.4f}% over {len(scc.cycle)-1} hops)  {path}")
        last_sig = cur_sig
        state.real_arb.append(new_row)

        if render_table:
            print("\033[H\033[J", end="")
            right = [f"OU: {state.status}",
                     f"TRUE ARBITRAGE  lambda/hop > tau={tau*100:+.4f}%  ({args.interval}s tick)",
                     "-" * 20] + (list(state.det_log) if state.det_log else ["(none clearing tau)"])
            print(feed._render_side_by_side(feed._exchange_rate_box(18), right))
        elif not args.headless_quiet:
            flag = "  <-- ARB PREDICTED" if (p and p.predicted_arb) else ""
            print(f"[{time.strftime('%H:%M:%S')}] lambda={lam*100:+.4f}%/hop{flag}", flush=True)

        if args.seconds and (now - state.t0) >= args.seconds:
            state.stop.set()
            break
        sl = args.interval - (time.monotonic() - now)
        if sl > 0:
            time.sleep(sl)


# ===========================================================================
# GUI backend selection (WSLg -> wayland)
# ===========================================================================
def _select_gui_backend():
    import matplotlib
    if os.environ.get("WAYLAND_DISPLAY") and "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "wayland"
    last_err = None
    for backend in ("QtAgg", "TkAgg"):
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as plt
            fig = plt.figure()
            fig.canvas.draw()
            plt.close(fig)
            return plt
        except Exception as e:
            last_err = e
    print(f"[no GUI backend] {last_err}\n"
          f"install one:  pip install PyQt5   (or: sudo apt install python3-tk)")
    return None


# ===========================================================================
# The pop-up window -- the SAME two-panel figure as ou_backtest.png, live
# ===========================================================================
def run_popup(feed, state: LiveState, args) -> None:
    plt = _select_gui_backend()
    if plt is None:
        print("falling back to headless (add --save for a GIF/PNG).")
        state.stop.wait()
        return
    print("opening OU popup window on your desktop (close it, or Ctrl-C here, to stop)...",
          flush=True)
    from matplotlib.animation import FuncAnimation

    c = _colors()
    tau = fee_threshold(args.fee)
    W, H = args.window, args.horizon

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), height_ratios=[3, 1])
    try:
        fig.canvas.manager.set_window_title("L4 -- OU tropical-eigenvalue arbitrage monitor")
    except Exception:
        pass

    def draw(_frame):
        # Snapshot the shared lists (GIL makes each read atomic); trim to a common len.
        xs = np.asarray(state.times[:], dtype=float)
        ys = np.asarray(state.lams[:], dtype=float)
        pm = np.asarray(state.pred_mean[:], dtype=float)
        pu = np.asarray(state.pred_upper[:], dtype=float)
        pa = np.asarray(state.pred_arb[:], dtype=bool)
        lm = np.asarray(state.lead_mean[:], dtype=float)
        ll = np.asarray(state.lead_lower[:], dtype=float)
        lu = np.asarray(state.lead_upper[:], dtype=float)
        ra = np.asarray(state.real_arb[:], dtype=bool)   # tick added a table row
        n = min(len(xs), len(ys), len(pm), len(pu), len(pa), len(lm), len(ll), len(lu), len(ra))
        ax.clear(); ax2.clear()
        if n == 0:
            ax.set_title("warming up the feed...")
            return
        xs, ys, pm, pu, pa = xs[:n], ys[:n], pm[:n], pu[:n], pa[:n]
        lm, ll, lu, ra = lm[:n], ll[:n], lu[:n], ra[:n]
        lo = max(0, n - W)
        tw, xw, pmw, puw, paw = xs[lo:], ys[lo:], pm[lo:], pu[lo:], pa[lo:]
        lmw, llw, luw, raw = lm[lo:], ll[lo:], lu[lo:], ra[lo:]
        dt = args.interval
        lead = args.lead
        PCT = 100.0                                  # per-hop log-return -> % per hop
        tau_p = tau * PCT

        # ---- top panel: eigenvalue + OU forecast (history + forward fan), in % ----
        ax.plot(tw, xw * PCT, color=c["actual"], lw=1.5, label="tropical eigenvalue lambda(t)")

        # The 1-step forecast made AT tick i is a prediction FOR tick i+1, so we plot
        # it at time t+dt (the instant it predicts), NOT at t. On a plateaued series a
        # 1-step forecast is ~"next = now" and would sit exactly on the actual line if
        # drawn at t -- looking suspiciously perfect. Shifted to t+dt, its one-tick LAG
        # is visible at every jump: at a step change the dashed line clings to the old
        # level for one tick before catching up. That gap IS the forecast error.
        finite = np.isfinite(pmw)
        if finite.any():
            ax.plot((tw + dt)[finite], (pmw * PCT)[finite], color=c["forecast"], lw=1.0,
                    ls="--", label="OU 1-step forecast (for t+1)")
            ax.fill_between(tw + dt, pmw * PCT, puw * PCT, where=finite, color=c["band"],
                            alpha=0.30, label="OU 95% band")

        # ---- the LEADING forecast: OU's call for lambda `lead` seconds later, drawn at
        # x=t+lead so it runs AHEAD of the live tape and keeps extending each frame. Its
        # right end is the pure-future prediction; where it overlaps the actual line it is
        # "what OU predicted `lead`s ago for now" -- watch reality arrive under it.
        lead_ok = np.isfinite(lmw)
        if lead_ok.any():
            ax.plot((tw + lead)[lead_ok], (lmw * PCT)[lead_ok], color="#9467bd", lw=1.7,
                    label=f"OU forecast (+{lead:.0f}s ahead)")
            ax.fill_between(tw + lead, llw * PCT, luw * PCT, where=lead_ok, color="#9467bd",
                            alpha=0.15)

        title = "OU rolling window  |  warming up..."
        if len(xw) >= args.min_fit:
            params = OUProcess.fit(xw, dt=dt)
            hl = params.half_life
            hl_s = f"{hl:.1f}s" if np.isfinite(hl) else "inf"
            title = (f"OU rolling window  |  theta={params.theta:.3f}  "
                     f"mu={params.mu * PCT:+.4f}%  half-life={hl_s}")
            # The whole efficient-market thesis in one line: OU pulls lambda toward its
            # equilibrium mu, which sits at/just-below 0 (arbitrage competed away). Draw
            # it so the "reverts to ~0" story is visible, not just asserted in the title.
            ax.axhline(params.mu * PCT, color=c["forecast"], ls="-.", lw=1.0, alpha=0.7,
                       label=f"OU equilibrium mu ({params.mu * PCT:+.4f}% ~ no-arb)")

        # GREEN o = a real arbitrage loop the TABLE logged this tick (new distinct loop),
        # so every green ring is one row in the arb table -- not one per tick lambda>tau.
        real = raw
        if real.any():
            ax.scatter(tw[real], (xw * PCT)[real], marker="o", facecolors="none",
                       edgecolors=c["real"], s=45, lw=1.3, zorder=4, label="real arb loop (table row)")
        # RED x = OU PREDICTED an arb -- shown at the loop ONSET (first tick of a run) so it
        # reads symmetrically with green: a predicted opportunity opening vs a real one.
        pa_prev = np.concatenate([[False], paw[:-1]]) if paw.size else paw
        predm = paw & finite & ~pa_prev
        if predm.any():
            ax.scatter((tw + dt)[predm], (pmw * PCT)[predm], marker="x", color=c["arb"],
                       s=55, lw=1.8, zorder=5, label="OU predicted arb (onset)")
        ax.axhline(tau_p, color=c["thresh"], ls=":", lw=1.2,
                   label=f"fee threshold tau ({args.fee * PCT:.3f}%/hop)")
        ax.axhline(0.0, color="#bbbbbb", lw=0.8)

        allv = np.concatenate([xw * PCT, (pmw * PCT)[finite], (puw * PCT)[finite],
                               (lmw * PCT)[lead_ok], (luw * PCT)[lead_ok], (llw * PCT)[lead_ok],
                               [tau_p, 0.0]])
        ymin, ymax = float(np.nanmin(allv)), float(np.nanmax(allv))
        pad = max((ymax - ymin) * 0.15, 1e-3)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_ylabel("per-hop return  (%)")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(alpha=0.2)

        # ---- bottom panel: cumulative per-hop edge. FIRE ONCE per predicted-arb
        # ONSET -- exactly the red-x drawn above -- NOT on every tick the prediction
        # stays true. The old code fired on all of `paw`, which (once mu>tau makes the
        # forecast a permanent buy) is silently an always-in-market strategy that
        # "captures" spikes it never forecast, while the chart shows only a couple of
        # markers. A persistence baseline (fire whenever lambda clears tau NOW, no
        # model) is overlaid on the SAME post-warmup ticks, so OU's line only looks
        # good if it actually beats the dashed one.
        if len(xw) >= 2:
            entry = predm[:-1]                              # one entry per red-x onset
            edge = np.where(entry, xw[1:] - tau, 0.0)
            cum = np.concatenate([[0.0], np.cumsum(edge)]) * PCT
            base_entry = (xw[:-1] > tau) & finite[:-1]      # persistence: arb-now -> fire
            base_edge = np.where(base_entry, xw[1:] - tau, 0.0)
            base_cum = np.concatenate([[0.0], np.cumsum(base_edge)]) * PCT
            ax2.plot(tw, cum, color="#333333", lw=1.3,
                     label=f"OU red-x entries (net {cum[-1]:+.4f}%)")
            ax2.plot(tw, base_cum, color="#8c564b", lw=1.0, ls="--",
                     label=f"persistence baseline (net {base_cum[-1]:+.4f}%)")
            ax2.axhline(0.0, color="#bbbbbb", lw=0.8)
            ax2.fill_between(tw, 0, cum, where=(cum >= 0), color=c["real"], alpha=0.3)
            ax2.fill_between(tw, 0, cum, where=(cum < 0), color=c["arb"], alpha=0.3)
            ax2.legend(loc="upper left", fontsize=7)
            ax2.set_title("cumulative per-hop edge: OU red-x entries vs persistence baseline",
                          fontsize=9)
        ax2.set_xlabel("time (s since start)")
        ax2.set_ylabel("cum. edge (%)")
        ax2.grid(alpha=0.2)

    interval_ms = max(150, min(1000, int(args.interval * 500)))  # smooth redraw, decoupled from sampling
    anim = FuncAnimation(fig, draw, interval=interval_ms, cache_frame_data=False)
    fig.canvas.mpl_connect("close_event", lambda _e: state.stop.set())
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        state.stop.set()


# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Live MultiVenueFeed table + OU popup graph, both off one shared "
                    "sampler. Defaults to a REALISTIC regime (anti-phantom guards on); "
                    "use --loose for the old theoretical-arb demo.")
    # 1s/tick is a good balance: fast enough to feel live, slow enough that lambda
    # drifts between ticks (0.5s plateaus). The forecast is drawn at t+dt so its lag
    # at jumps stays visible even when the tape is calm. Raise --interval if you want
    # more drift per tick, lower it if you want a snappier refresh.
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between samples (table+graph tick)")
    ap.add_argument("--window", type=int, default=90, help="rolling-window length (samples)")
    ap.add_argument("--min-fit", type=int, default=12, help="samples before OU calibrates")
    ap.add_argument("--horizon", type=int, default=20, help="forecast look-ahead (steps)")
    ap.add_argument("--lead", type=float, default=6.0,
                    help="seconds the OU forecast line runs AHEAD of the live tape (the leading prediction)")
    ap.add_argument("--fee", type=float, default=0.00023,
                    help="arb threshold tau: per-hop fee a loop must clear (the red-x line)")
    ap.add_argument("--feed-fee", type=float, default=0.00015,
                    help="per-leg taker fee baked into graph edges (0 = frictionless)")
    ap.add_argument("--quote-window", type=float, default=0.2, help="max seconds apart cycle legs may be")
    ap.add_argument("--max-quote-age", type=float, default=1.0, help="drop quotes older than this")
    ap.add_argument("--min-profit", type=float, default=0.0, help="table: hide loops under this net return")
    ap.add_argument("--no-depth-filter", action="store_true",
                    help="disable the min-notional depth filter (lets thin-book phantom arb through)")
    ap.add_argument("--loose", action="store_true",
                    help="low-friction DEMO regime: frictionless edges, wide 5s windows, depth "
                         "filter OFF -- surfaces theoretical/phantom arb that is NOT tradeable")
    ap.add_argument("--strict", action="store_true",
                    help="tighter realistic regime: fee=0.2%%, feed-fee=0.03%%, tight windows, depth on")
    ap.add_argument("--no-feed-view", action="store_true", help="skip the terminal table")
    ap.add_argument("--headless", action="store_true", help="no popup; sample + optionally --save")
    ap.add_argument("--seconds", type=float, default=0.0, help="auto-stop after N seconds")
    ap.add_argument("--save", action="store_true", help="render GIF+PNG of the live tape on exit")
    args = ap.parse_args()
    args.headless_quiet = False

    # Default is the REALISTIC regime (guards on) so lambda is tradeable and the red-x
    # onsets stay visible. --loose re-opens every guard (old demo behaviour: phantom
    # arb, lambda pinned high above tau); --strict tightens fee/windows further. If both
    # are passed, --strict wins.
    if args.loose:
        args.feed_fee = 0.0
        args.quote_window, args.max_quote_age = 5.0, 5.0
        args.no_depth_filter = True
    if args.strict:
        args.fee, args.feed_fee = 0.002, 0.0003
        args.quote_window, args.max_quote_age = 0.2, 1.0
        args.no_depth_filter = False

    if not args.headless and not os.environ.get("DISPLAY") \
            and not os.environ.get("WAYLAND_DISPLAY") and sys.platform != "darwin":
        print("no display detected -- falling back to --headless (use --save to get a GIF/PNG).")
        args.headless = True

    feed_kwargs = dict(fee=args.feed_fee, max_quote_age=args.max_quote_age,
                       quote_window=args.quote_window)
    if args.no_depth_filter:
        feed_kwargs["min_notional"] = None
    feed = build_default_feed(**feed_kwargs)
    regime = "STRICT" if args.strict else ("LOOSE/phantom" if args.loose else "REALISTIC")
    edges = "frictionless" if args.feed_fee == 0 else f"{args.feed_fee*100:.3f}%/leg"
    print(f"[{regime}] arb fee tau={args.fee*100:.3f}%/hop  edges={edges}  "
          f"quote_window={args.quote_window}s  depth_filter={'OFF' if args.no_depth_filter else 'on'}",
          flush=True)

    state = LiveState()
    predictor = RollingArbitragePredictor(
        window=args.window, dt=args.interval, fee=args.fee, min_fit=args.min_fit
    )

    render_table = (not args.no_feed_view) and (not args.headless)
    sampler = threading.Thread(
        target=sampler_loop, args=(feed, state, args, predictor, render_table), daemon=True
    )
    sampler.start()

    try:
        if args.headless:
            print(f"[headless] sampling every {args.interval}s "
                  f"{'for %.0fs' % args.seconds if args.seconds else 'until Ctrl-C'}...", flush=True)
            state.stop.wait()
        else:
            run_popup(feed, state, args)
    except KeyboardInterrupt:
        pass
    finally:
        state.stop.set()
        for _, dashboard in feed.dashboards:
            dashboard.is_running = False
        time.sleep(0.2)

    if args.save and len(state.lams) > args.min_fit:
        t_arr = np.asarray(state.times)
        lam_arr = np.asarray(state.lams)
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
