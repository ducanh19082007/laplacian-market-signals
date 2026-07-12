"""
L4 -- REGIME & RISK ENGINE
==========================

This layer does NOT try to predict arbitrage. We established (and it is provable on
the tape) that the tropical eigenvalue at a 1 s tick is white-noise-plus-jumps with
~zero one-step autocorrelation, and that any opening reverts in << the round-trip
latency -- so a *tradeable* forecast is impossible at this observation scale. That is
the market being efficient, not the pipeline failing.

So L4 is re-pointed at a question that IS well-posed: measure the market's REGIME and
RISK from two spectra of the live asset x venue graph, every tick:

  * tropical (max-plus) eigenvalue  lambda   -> ARBITRAGE INTENSITY
        the best per-hop cycle return: how mispriced the market is right now.
  * Fiedler value (Laplacian lambda_2)       -> CONNECTIVITY
        algebraic connectivity: how tightly the venues/assets are coupled. HIGH =
        one well-arbitraged market; a DROP = the graph fragmenting (venues/assets
        decoupling -> liquidity stress). Unlike the arb spikes this is PERSISTENT,
        so it is actually usable at 1 Hz.
  * MarketStrain = 1/lambda_2                 -> fragmentation pressure.

Each tick is classified into a regime:

    FRAGMENTING  the graph has split into >= 2 pieces (or lambda_2 ~ 0) -- venues
                 decoupling / liquidity withdrawing. The top risk state; overrides.
    STRESSED     still connected, but arb intensity lambda >> the fee -- real, large
                 dislocations are open (this is the only regime where arb is real).
    EFFICIENT    connected and quiet -- lambda near/under the fee, the normal state.

Why component-count and not just lambda_2: on the full 45-asset universe a single thin
alt with no fresh cross-quote is an isolated node, which pins lambda_2 at 0 by itself.
So we read the WHOLE Laplacian spectrum: the multiplicity of eigenvalue 0 is the number
of disconnected pieces (the real fragmentation signal), and the smallest NON-zero
eigenvalue is how tightly the connected part holds together.

Run (from the repo root):
    python "L4_Regime&RiskEngine/regime_engine.py"                 # live, REALISTIC feed
    python "L4_Regime&RiskEngine/regime_engine.py" --headless --seconds 60
    python "L4_Regime&RiskEngine/regime_engine.py" --demo          # offline, no feed
    python "L4_Regime&RiskEngine/regime_engine.py" --loose         # theoretical-arb feed

Author: Anh Duc Le
"""

import argparse
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from L1_DataProcessing.MultiVenueFeed import build_default_feed
from L1_DataProcessing.DataProcessing import ExchangeRateGraph
from L2_MarketStructureAnalysis.TropicalEigenvalue import TropicalEigenvalue, fee_threshold
from L2_MarketStructureAnalysis.GraphLaplacian import Laplacian


# ===========================================================================
# Regime labels + palettes
# ===========================================================================
EFFICIENT = "EFFICIENT"
STRESSED = "STRESSED"
FRAGMENTING = "FRAGMENTING"
WARMING = "WARMING"

_ANSI = {EFFICIENT: "\033[92m", STRESSED: "\033[93m", FRAGMENTING: "\033[91m", WARMING: "\033[2m"}
_RST = "\033[0m"
_RGB = {EFFICIENT: "#2ca02c", STRESSED: "#ff7f0e", FRAGMENTING: "#d62728", WARMING: "#999999"}


# ===========================================================================
# Core: read the two spectra off one graph and classify the regime
# ===========================================================================
@dataclass
class RegimeReading:
    t: float
    lam: float            # tropical eigenvalue: arb intensity (per-hop log-return); NaN if no cycle
    fiedler: float        # Fiedler value lambda_2: algebraic connectivity (0 if graph split)
    connectivity: float   # smallest NON-zero Laplacian eigenvalue (tightness of the connected part)
    strain: float         # 1/lambda_2 (inf if split)
    n_components: int     # disconnected pieces (1 == fully connected)
    n_nodes: int
    regime: str
    top_loop: str = ""    # the loop carrying lambda, for context in the table


def spectral_read(g) -> tuple:
    """
    Full normalized-Laplacian spectrum off the live graph -> connectivity metrics.

    Returns (fiedler, connectivity, n_components, n_nodes):
      fiedler       lambda_2 (second-smallest eigenvalue); 0 when the graph is split
      connectivity  smallest eigenvalue > 0 -- how tightly the connected part holds
      n_components  multiplicity of eigenvalue 0 == number of disconnected pieces
    (nan, nan, 0, n) on a degenerate graph.
    """
    try:
        # 'unweighted' (binary presence) is the author's recommended default AND the only
        # correct mode here: zero-cost same-asset transfer edges across venues have weight
        # -log(1)=0, so a magnitude-weighted ('average') affinity drops them and spuriously
        # splits the market by venue. Presence-based keeps the true connectivity topology.
        lap = Laplacian(g, attr="weight", symmetrize="unweighted")
        n = lap.n
        if n < 2:
            return (np.nan, np.nan, max(n, 0), n)
        L = lap.NormalizedLaplacian()
        evals = np.clip(np.linalg.eigvalsh(L), 0.0, None)   # ascending; clip fp noise
        n_comp = int(np.sum(evals < 1e-8))
        fiedler = float(evals[1])
        nz = evals[evals >= 1e-8]
        connectivity = float(nz.min()) if nz.size else 0.0
        return (fiedler, connectivity, n_comp, n)
    except Exception:
        return (np.nan, np.nan, 0, 0)


def classify(lam: float, n_components: int, fiedler: float, tau: float,
             *, stress_mult: float, frag_components: int = 2) -> str:
    """Regime from arb intensity + connectivity. Fragmentation risk overrides."""
    if n_components is None or n_components >= frag_components or not np.isfinite(fiedler):
        return FRAGMENTING
    if np.isfinite(lam) and lam > stress_mult * tau:
        return STRESSED
    return EFFICIENT


def read_regime(g, fee: float, t: float = 0.0, *, stress_mult: float = 5.0) -> RegimeReading:
    """One graph -> its regime reading (both spectra + label)."""
    tau = fee_threshold(fee)
    res = TropicalEigenvalue(g).compute()
    lam = float(res.eigenvalue)
    if not np.isfinite(lam):
        lam = np.nan                                        # no cycle => no arbitrage
    fiedler, connectivity, n_comp, n_nodes = spectral_read(g)
    strain = (1.0 / fiedler) if (np.isfinite(fiedler) and fiedler > 1e-12) else np.inf
    regime = classify(lam, n_comp, fiedler, tau, stress_mult=stress_mult)
    top = ""
    if getattr(res, "has_cycle", False) and getattr(res, "cycle", None):
        top = " -> ".join(ExchangeRateGraph.fmt(x) for x in res.cycle)
    return RegimeReading(t=t, lam=lam, fiedler=fiedler, connectivity=connectivity,
                         strain=strain, n_components=n_comp, n_nodes=n_nodes,
                         regime=regime, top_loop=top)


# ===========================================================================
# Shared state (single source of truth for terminal + popup)
# ===========================================================================
class RegimeState:
    def __init__(self) -> None:
        self.times: list = []
        self.lams: list = []
        self.fiedlers: list = []
        self.strains: list = []
        self.regimes: list = []
        self.log: deque = deque(maxlen=15)      # regime-transition log for the table
        self.status: str = "warming up..."
        self.stop = threading.Event()
        self.t0 = time.monotonic()
        self._last_regime = None


def _fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}" if np.isfinite(x) else ("inf" if x == np.inf else "n/a")


# ===========================================================================
# The one sampler: build graph -> two spectra -> regime -> (optionally) table
# ===========================================================================
def sampler_loop(feed, state: RegimeState, args, render_table: bool) -> None:
    tau = fee_threshold(args.fee)
    while not state.stop.is_set():
        now = time.monotonic()
        g = feed.build_graph()
        if g is None:
            time.sleep(args.interval)
            continue
        r = read_regime(g, args.fee, t=now - state.t0, stress_mult=args.stress_mult)
        if r.n_nodes < 2:                       # graph still warming / too small to score
            time.sleep(args.interval)
            continue

        state.times.append(r.t)
        state.lams.append(r.lam)
        state.fiedlers.append(r.fiedler)
        state.strains.append(r.strain)
        state.regimes.append(r.regime)

        if r.regime != state._last_regime:      # log only on a regime CHANGE
            state.log.append(
                f"[{time.strftime('%H:%M:%S')}] -> {r.regime}   "
                f"lambda={_fmt(r.lam * 100)}%/hop  lambda2={_fmt(r.fiedler)}  "
                f"comps={r.n_components}" + (f"  {r.top_loop}" if r.top_loop else ""))
            state._last_regime = r.regime

        state.status = (f"REGIME={r.regime}  lambda={_fmt(r.lam * 100)}%/hop  "
                        f"lambda2={_fmt(r.fiedler)}  strain={_fmt(r.strain, 1)}  "
                        f"comps={r.n_components}/{r.n_nodes}n")

        if render_table:
            print("\033[H\033[J", end="")
            col = _ANSI.get(r.regime, "")
            right = [
                f"L4 REGIME & RISK ENGINE   ({args.interval}s tick)",
                f"  current regime : {col}{r.regime}{_RST}",
                f"  arb intensity  : lambda  = {_fmt(r.lam * 100)}%/hop  (stress> {tau*args.stress_mult*100:.4f}%)",
                f"  connectivity   : lambda2 = {_fmt(r.fiedler)}   strain = {_fmt(r.strain, 1)}",
                f"  graph          : {r.n_nodes} nodes / {r.n_components} component(s)",
                "-" * 20, "REGIME CHANGES:"] + (list(state.log) or ["(none yet)"])
            print(feed._render_side_by_side(feed._exchange_rate_box(16), right))
        elif not getattr(args, "headless_quiet", False):
            print(f"[{time.strftime('%H:%M:%S')}] {r.regime:11s}  lambda={_fmt(r.lam * 100)}%/hop  "
                  f"lambda2={_fmt(r.fiedler)}  strain={_fmt(r.strain, 1)}  comps={r.n_components}",
                  flush=True)

        if args.seconds and (now - state.t0) >= args.seconds:
            state.stop.set()
            break
        sl = args.interval - (time.monotonic() - now)
        if sl > 0:
            time.sleep(sl)


# ===========================================================================
# GUI backend selection (WSLg -> wayland), same pattern as the rest of the repo
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
# The pop-up: regime time-series (top) + the 2-D regime MAP (bottom)
# ===========================================================================
def run_popup(feed, state: RegimeState, args) -> None:
    plt = _select_gui_backend()
    if plt is None:
        print("falling back to headless.")
        state.stop.wait()
        return
    print("opening the regime monitor window (close it, or Ctrl-C here, to stop)...", flush=True)
    from matplotlib.animation import FuncAnimation
    from matplotlib.lines import Line2D

    tau = fee_threshold(args.fee)
    stress_y = tau * args.stress_mult * 100.0
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), height_ratios=[3, 2])
    axr = ax.twinx()
    try:
        fig.canvas.manager.set_window_title("L4 -- Regime & Risk Engine")
    except Exception:
        pass

    legend_handles = [Line2D([0], [0], marker="o", ls="", color=_RGB[k], label=k)
                      for k in (EFFICIENT, STRESSED, FRAGMENTING)]

    def draw(_frame):
        xs = np.asarray(state.times[:], dtype=float)
        lam = np.asarray(state.lams[:], dtype=float)
        fie = np.asarray(state.fiedlers[:], dtype=float)
        reg = list(state.regimes[:])
        n = min(len(xs), len(lam), len(fie), len(reg))
        ax.clear(); axr.clear(); ax2.clear()
        if n == 0:
            ax.set_title("warming up the feed...")
            return
        xs, lam, fie, reg = xs[:n], lam[:n], fie[:n], reg[:n]
        cols = [_RGB.get(r, "#999999") for r in reg]

        # ---- top: arb intensity lambda (left) + connectivity lambda2 (right), points
        #      colored by regime so the state over time is legible at a glance.
        ax.plot(xs, lam * 100, color="#1f77b4", lw=1.0, alpha=0.6,
                label="arb intensity lambda (%/hop)")
        ax.scatter(xs, lam * 100, c=cols, s=14, zorder=3)
        ax.axhline(stress_y, color="#d62728", ls=":", lw=1.0,
                   label=f"stress threshold ({stress_y:.4f}%/hop)")
        ax.axhline(0.0, color="#bbbbbb", lw=0.8)
        ax.set_ylabel("arb intensity  (%/hop)")
        ax.set_xlabel("time (s since start)")
        axr.plot(xs, fie, color="#9467bd", lw=1.4, label="connectivity lambda2")
        axr.set_ylabel("connectivity  lambda2", color="#9467bd")
        axr.tick_params(axis="y", labelcolor="#9467bd")
        cur = reg[-1]
        ax.set_title(f"current regime: {cur}   |   lambda={_fmt(lam[-1]*100)}%/hop   "
                     f"lambda2={_fmt(fie[-1])}   comps drive FRAGMENTING")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.2)

        # ---- bottom: the 2-D REGIME MAP. Where the market lives in
        #      (connectivity, arb-intensity) space; the last point is where it is NOW.
        ax2.scatter(fie, lam * 100, c=cols, s=20, alpha=0.7)
        ax2.scatter(fie[-1], lam[-1] * 100, s=160, facecolors="none",
                    edgecolors="black", lw=1.6, zorder=5, label="now")
        ax2.axhline(stress_y, color="#d62728", ls=":", lw=1.0)     # above => STRESSED
        ax2.axvline(0.02, color="#9467bd", ls=":", lw=1.0)         # near 0 => FRAGMENTING
        ax2.set_xlabel("connectivity  lambda2  (low => fragmenting)")
        ax2.set_ylabel("arb intensity (%/hop)")
        ax2.set_title("regime map")
        ax2.legend(handles=legend_handles + [Line2D([0], [0], marker="o", ls="",
                    markerfacecolor="none", markeredgecolor="black", label="now")],
                   loc="upper right", fontsize=8)
        ax2.grid(alpha=0.2)

    interval_ms = max(150, min(1000, int(args.interval * 500)))
    anim = FuncAnimation(fig, draw, interval=interval_ms, cache_frame_data=False)
    fig.canvas.mpl_connect("close_event", lambda _e: state.stop.set())
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        state.stop.set()


# ===========================================================================
# Offline demo -- no feed, so the classifier is verifiable anywhere
# ===========================================================================
def _demo_graphs():
    """Three hand-built snapshots, one per regime (no network)."""
    def G(assets, snap):
        return ExchangeRateGraph(assets, transfer_cost=0.0).build_from_snapshot(snap).log_transform()

    tri = ["btc", "eth", "usdt"]
    efficient = {
        "btcusdt": {"Binance": {"bid": "50000", "ask": "50001"}, "Kraken": {"bid": "49999", "ask": "50000"}},
        "ethusdt": {"Binance": {"bid": "3000.0", "ask": "3000.1"}, "Kraken": {"bid": "2999.9", "ask": "3000.0"}},
        "ethbtc":  {"Binance": {"bid": "0.05999", "ask": "0.06001"}, "Kraken": {"bid": "0.05998", "ask": "0.06000"}},
    }
    # ethbtc badly mispriced vs the usdt legs -> a real fee-clearing loop, still connected.
    stressed = {
        "btcusdt": {"Binance": {"bid": "50000", "ask": "50001"}, "Kraken": {"bid": "49999", "ask": "50000"}},
        "ethusdt": {"Binance": {"bid": "3000.0", "ask": "3000.1"}, "Kraken": {"bid": "2999.9", "ask": "3000.0"}},
        "ethbtc":  {"Binance": {"bid": "0.0630", "ask": "0.0631"}, "Kraken": {"bid": "0.0629", "ask": "0.0630"}},
    }
    # two disjoint clusters ({btc,usdt} and {xrp,eth}) with no shared asset -> a genuinely
    # disconnected graph (>= 2 components): the real fragmentation case.
    frag_assets = ["btc", "usdt", "xrp", "eth"]
    fragmenting = {
        "btcusdt": {"Binance": {"bid": "50000", "ask": "50001"}, "Kraken": {"bid": "49999", "ask": "50000"}},
        "xrpeth":  {"Binance": {"bid": "0.00019", "ask": "0.00020"}, "Kraken": {"bid": "0.00019", "ask": "0.00020"}},
    }
    return [("efficient", G(tri, efficient)), ("stressed", G(tri, stressed)),
            ("fragmenting", G(frag_assets, fragmenting))]


def run_demo(fee: float, stress_mult: float) -> None:
    print("=" * 66)
    print("L4 REGIME & RISK ENGINE -- offline demo (no feed)")
    print(f"fee tau = {fee*100:.4f}%/hop   stress threshold = {fee_threshold(fee)*stress_mult*100:.4f}%/hop")
    print("-" * 66)
    for name, g in _demo_graphs():
        r = read_regime(g, fee, stress_mult=stress_mult)
        print(f"[{name:>11s} snapshot] -> {r.regime:11s}  "
              f"lambda={_fmt(r.lam*100)}%/hop  lambda2={_fmt(r.fiedler)}  "
              f"strain={_fmt(r.strain,1)}  {r.n_nodes} nodes / {r.n_components} comp(s)")
        if r.top_loop:
            print(f"                          top loop: {r.top_loop}")
    print("=" * 66)


# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="L4 Regime & Risk Engine: classify the market regime each tick from "
                    "the tropical eigenvalue (arb intensity) and Fiedler value (connectivity).")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    ap.add_argument("--fee", type=float, default=0.00023, help="per-hop fee tau (sets the stress threshold)")
    ap.add_argument("--stress-mult", type=float, default=5.0,
                    help="STRESSED when arb intensity lambda > this * tau")
    # feed regime knobs -- same anti-phantom defaults as the rest of L4/L1.
    ap.add_argument("--feed-fee", type=float, default=0.00015, help="per-leg taker fee in graph edges")
    ap.add_argument("--quote-window", type=float, default=0.2, help="max seconds apart cycle legs may be")
    ap.add_argument("--max-quote-age", type=float, default=1.0, help="drop quotes older than this")
    ap.add_argument("--no-depth-filter", action="store_true", help="disable the min-notional depth filter")
    ap.add_argument("--loose", action="store_true",
                    help="low-friction DEMO feed: frictionless, wide 5s windows, depth off (phantom arb)")
    ap.add_argument("--strict", action="store_true",
                    help="tighter realistic feed: feed-fee=0.03%%, tight windows, depth on")
    ap.add_argument("--no-feed-view", action="store_true", help="skip the terminal table")
    ap.add_argument("--headless", action="store_true", help="no popup; sample + print regime")
    ap.add_argument("--seconds", type=float, default=0.0, help="auto-stop after N seconds")
    ap.add_argument("--demo", action="store_true", help="offline classifier demo (no feed) and exit")
    args = ap.parse_args()
    args.headless_quiet = False

    if args.demo:
        run_demo(args.fee, args.stress_mult)
        return

    if args.loose:
        args.feed_fee, args.quote_window, args.max_quote_age = 0.0, 5.0, 5.0
        args.no_depth_filter = True
    if args.strict:
        args.feed_fee, args.quote_window, args.max_quote_age = 0.0003, 0.2, 1.0
        args.no_depth_filter = False

    if not args.headless and not os.environ.get("DISPLAY") \
            and not os.environ.get("WAYLAND_DISPLAY") and sys.platform != "darwin":
        print("no display detected -- falling back to --headless.")
        args.headless = True

    feed_kwargs = dict(fee=args.feed_fee, max_quote_age=args.max_quote_age,
                       quote_window=args.quote_window)
    if args.no_depth_filter:
        feed_kwargs["min_notional"] = None
    feed = build_default_feed(**feed_kwargs)
    regime = "STRICT" if args.strict else ("LOOSE/phantom" if args.loose else "REALISTIC")
    print(f"[{regime} feed] stress> {fee_threshold(args.fee)*args.stress_mult*100:.4f}%/hop  "
          f"quote_window={args.quote_window}s  depth_filter={'OFF' if args.no_depth_filter else 'on'}",
          flush=True)

    state = RegimeState()
    render_table = (not args.no_feed_view) and (not args.headless)
    sampler = threading.Thread(target=sampler_loop, args=(feed, state, args, render_table), daemon=True)
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


if __name__ == "__main__":
    main()
