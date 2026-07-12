"""
L4 -- Regime & Risk Engine: OU model on the tropical eigenvalue + arbitrage
prediction and backtest.

WHAT THIS LAYER DOES
--------------------
L2's TropicalEigenvalue gives ONE number per market snapshot: the max-plus
eigenvalue lambda = the per-hop log-return of the single best loop in the graph
right now (see L2_MarketStructureAnalysis/TropicalEigenvalue.py). Watch that number
tick after tick and it becomes a TIME SERIES lambda(t).

The core claim (and the reason OU is the right model): an efficient market has NO
persistent free money. When a loop opens (lambda spikes up), other traders close it
almost immediately, so lambda is pulled back down toward an equilibrium that sits at
or just below zero. That "pulled back toward a level" behaviour is exactly an
Ornstein-Uhlenbeck process, NOT a random walk:

        dX = theta ( mu - X ) dt + sigma dW

    X (= lambda, the tropical eigenvalue)   the thing we track
    mu   ~ 0     the no-arbitrage equilibrium (efficient market => ~0, often slightly
                 negative once you subtract the fee it costs to trade)
    theta        mean-reversion SPEED: how fast an opened loop gets closed. Big theta
                 => arbitrage evaporates in a blink (half-life = ln2/theta).
    sigma        volatility of the eigenvalue.

So mapping the user's shorthand onto the standard OU:
    X_{n-1}  = the current tropical-max eigenvalue (the state we forecast FROM)
    theta    = theta(the eigenvalue series)  -- fit from the window
    mu ~ 0   = the equilibrium the loop reverts to

ROLLING WINDOW
--------------
We only ever fit on the last `window` seconds/ticks. As time advances, old points
fall off the left and new ones arrive on the right -- the OU parameters are always
calibrated to the CURRENT regime, and the plot is a window that scrolls.

ARBITRAGE FLAG (the red x)
-------------------------
A loop is net-profitable only if its per-hop return beats the per-hop taker fee:
    product*(1-fee)^L > 1  <=>  lambda > -ln(1-fee)  =: tau(fee)
tau(fee) is a horizontal threshold (from L2.fee_threshold). We forecast the next
eigenvalue with the OU model; if the forecast crosses tau we mark that instant with
a red x = "OU predicts an exploitable loop here". The BACKTEST then asks the honest
question: at those red-x instants, did a real, fee-clearing loop actually exist the
next tick -- and if you had fired, what did you net?

The "high fee" backtest is the whole point: raise tau and almost every predicted
loop stops clearing -- mean reversion is faster than you can trade, so the captured
edge collapses. That is the external friction the user asked to stress-test against.

Ref (Layer 4): Sidharth Mallik, "Pricing cryptocurrencies: Modelling the ETHBTC
spot-quotient variation as a diffusion process", arXiv:2111.11609 -- OU calibration
of a crypto quotient, Sec. 3.

Run it:   python "L4_Regime&RiskEngine/OUArbitrage.py"      (from the repo root)

Author: Anh Duc Le
"""

import math
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# This folder ('L4_Regime&RiskEngine') is NOT an importable package name (the '&'
# is illegal in an identifier), so we run as a script and put the repo root on the
# path to reach the real layers.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from L1_DataProcessing.DataProcessing import ExchangeRateGraph
from L2_MarketStructureAnalysis.TropicalEigenvalue import (
    TropicalEigenvalue,
    fee_threshold,
)


# ===========================================================================
# 1. OU calibration + forecast
# ===========================================================================
@dataclass
class OUParams:
    """Calibrated Ornstein-Uhlenbeck parameters on a fixed step dt."""
    theta: float        # mean-reversion speed (>0 => reverting)
    mu: float           # long-run mean (the no-arb equilibrium, ~0)
    sigma: float        # volatility of the driving noise
    dt: float           # sampling step the fit was done on

    @property
    def half_life(self) -> float:
        """Time for a deviation to shrink by half: ln2/theta. inf if not reverting."""
        return math.log(2.0) / self.theta if self.theta > 1e-12 else float("inf")

    @property
    def is_mean_reverting(self) -> bool:
        return self.theta > 1e-9

    @property
    def stationary_std(self) -> float:
        """Std of the OU stationary distribution: sigma/sqrt(2 theta)."""
        return self.sigma / math.sqrt(2.0 * self.theta) if self.theta > 1e-12 else float("inf")


class OUProcess:
    """
    Calibrate and forecast an OU process from a discretely-sampled series.

    Discretisation. Exact OU sampled every dt is the AR(1) recursion
        X_{t+1} = a + b X_t + eps,   b = e^{-theta dt},   a = mu (1 - b),
        Var(eps) = sigma^2 (1 - b^2) / (2 theta).
    So fitting OU == an ordinary least-squares regression of X_{t+1} on X_t, then
    inverting:
        theta = -ln(b)/dt,  mu = a/(1-b),  sigma = sqrt( Var(eps) * 2 theta/(1-b^2) ).
    """

    @staticmethod
    def fit(series: Sequence[float], dt: float = 1.0) -> OUParams:
        x = np.asarray(series, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 3:
            # Not enough to regress: degenerate, no reversion. mu = mean (or 0).
            mu = float(x.mean()) if x.size else 0.0
            return OUParams(theta=0.0, mu=mu, sigma=0.0, dt=dt)

        x0, x1 = x[:-1], x[1:]
        # OLS slope/intercept of x1 = a + b x0.
        x0m, x1m = x0.mean(), x1.mean()
        var0 = float(((x0 - x0m) ** 2).sum())
        if var0 <= 1e-18:
            # x0 is constant => no dynamics to learn.
            return OUParams(theta=0.0, mu=float(x1m), sigma=float(x1.std()), dt=dt)
        b = float(((x0 - x0m) * (x1 - x1m)).sum() / var0)
        a = float(x1m - b * x0m)

        # b must be in (0,1) for a genuine, non-oscillating mean reversion. Clamp so
        # theta stays finite and positive; a b>=1 means "not reverting on this window".
        b_clamped = min(max(b, 1e-6), 1.0 - 1e-9)
        theta = -math.log(b_clamped) / dt
        mu = a / (1.0 - b_clamped)

        resid = x1 - (a + b * x0)
        resid_var = float((resid ** 2).mean())
        denom = 1.0 - b_clamped ** 2
        sigma = math.sqrt(max(resid_var, 0.0) * (2.0 * theta) / denom) if denom > 1e-12 else float(np.sqrt(resid_var))

        # If the regression says b >= 1 (trending / random walk on this window), report
        # theta ~ 0 so callers can see "not reverting" via is_mean_reverting.
        if b >= 1.0:
            theta = 0.0
        return OUParams(theta=theta, mu=mu, sigma=sigma, dt=dt)

    @staticmethod
    def forecast(
        params: OUParams, x_last: float, horizon: int, z: float = 1.96
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forecast `horizon` steps ahead from x_last. Closed-form OU moments:
            E[X_{t+k}]   = mu + (x_last - mu) e^{-theta k dt}
            Var[X_{t+k}] = sigma^2/(2 theta) (1 - e^{-2 theta k dt})
        Returns (mean, lower, upper) arrays of length `horizon`; the band is +/- z sd.
        """
        ks = np.arange(1, horizon + 1)
        tau = ks * params.dt
        if params.theta > 1e-12:
            decay = np.exp(-params.theta * tau)
            mean = params.mu + (x_last - params.mu) * decay
            var = (params.sigma ** 2) / (2.0 * params.theta) * (1.0 - np.exp(-2.0 * params.theta * tau))
        else:
            # No reversion => forecast is flat with random-walk-like growing variance.
            mean = np.full(horizon, x_last, dtype=float)
            var = (params.sigma ** 2) * tau
        sd = np.sqrt(np.maximum(var, 0.0))
        return mean, mean - z * sd, mean + z * sd


# ===========================================================================
# 2. Rolling predictor -- the live-facing object
# ===========================================================================
@dataclass
class Prediction:
    """One step's OU read: what we expect next and whether it's a (predicted) arb."""
    t: float
    x_now: float
    params: OUParams
    mean_next: float
    upper_next: float
    threshold: float
    predicted_arb: bool             # OU expects the next eigenvalue to clear the fee
    possible_arb: bool              # upper band clears the fee (optimistic)


class RollingArbitragePredictor:
    """
    Feed it (t, lambda) as they arrive; it keeps a rolling window, recalibrates OU on
    every push, and forecasts whether the NEXT tick holds a fee-clearing loop.
    """

    def __init__(
        self,
        window: int = 60,
        dt: float = 1.0,
        fee: float = 0.002,
        min_fit: int = 15,
        z: float = 1.96,
    ) -> None:
        self.window = window
        self.dt = dt
        self.fee = fee
        self.threshold = fee_threshold(fee)
        self.min_fit = min_fit
        self.z = z
        self.buf: Deque[Tuple[float, float]] = deque(maxlen=window)

    def push(self, t: float, lam: float) -> Optional[Prediction]:
        """Append a sample and return the fresh one-step prediction (None until warm)."""
        self.buf.append((t, lam))
        if len(self.buf) < self.min_fit:
            return None
        series = [v for _, v in self.buf]
        params = OUProcess.fit(series, dt=self.dt)
        mean, lower, upper = OUProcess.forecast(params, series[-1], horizon=1, z=self.z)
        pred = Prediction(
            t=t,
            x_now=lam,
            params=params,
            mean_next=float(mean[0]),
            upper_next=float(upper[0]),
            threshold=self.threshold,
            predicted_arb=float(mean[0]) > self.threshold,
            possible_arb=float(upper[0]) > self.threshold,
        )
        return pred

    def forecast_path(self, horizon: int = 20) -> Tuple[OUParams, np.ndarray, np.ndarray, np.ndarray]:
        """Multi-step forecast fan from the current window (for the plot's look-ahead)."""
        series = [v for _, v in self.buf]
        params = OUProcess.fit(series, dt=self.dt)
        mean, lower, upper = OUProcess.forecast(params, series[-1], horizon=horizon, z=self.z)
        return params, mean, lower, upper


# ===========================================================================
# 3. Feeding the series -- live hook + offline replay
# ===========================================================================
def eigenvalue_from_graph(graph: "ExchangeRateGraph") -> float:
    """LIVE HOOK: one snapshot's graph -> its tropical-max eigenvalue (L2)."""
    return TropicalEigenvalue(graph).eigenvalue()


def series_from_snapshots(
    snapshots: Iterable[dict],
    assets: Sequence[str],
    fee: float = 0.0,
    transfer_cost: float = 0.0,
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    OFFLINE REPLAY: turn a sequence of raw snapshot dicts (the shape
    MultiBrokerOrderBook.snapshot() emits) into (times, eigenvalues). This is the
    real path L4 would use on a recorded feed; the synthetic simulator below stands
    in until a recording exists.
    """
    ts: List[float] = []
    lams: List[float] = []
    for i, snap in enumerate(snapshots):
        g = ExchangeRateGraph(assets, transfer_cost=transfer_cost, fee=fee).build_from_snapshot(snap)
        lams.append(eigenvalue_from_graph(g))
        ts.append(i * dt)
    return np.asarray(ts), np.asarray(lams)


def simulate_eigenvalue_series(
    n: int = 400,
    dt: float = 1.0,
    theta: float = 0.35,
    mu: float = -0.0008,
    sigma: float = 0.0016,
    jump_rate: float = 0.05,
    jump_size: float = 0.010,
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SYNTHETIC series that behaves like a real tropical-eigenvalue tape so the whole
    pipeline is demonstrable with no feed:
      * a base OU pulled to mu ~ 0^- (efficient market, slightly below zero after
        the cost of trading),
      * plus occasional POSITIVE jumps (a loop opens) that the OU pull then closes
        over the next few ticks -- exactly the transient arbitrage we want to catch.
    Returns (times, eigenvalues). Deterministic given `seed`.
    """
    rng = np.random.default_rng(seed)
    x = np.empty(n, dtype=float)
    x[0] = mu
    b = math.exp(-theta * dt)
    eps_sd = sigma * math.sqrt((1 - b * b) / (2 * theta)) if theta > 0 else sigma
    for i in range(1, n):
        x[i] = mu + b * (x[i - 1] - mu) + rng.normal(0.0, eps_sd)
        if rng.random() < jump_rate:                     # a loop suddenly opens
            x[i] += abs(rng.normal(jump_size, jump_size * 0.5))
    t = np.arange(n) * dt
    return t, x


# ===========================================================================
# 4. Backtest -- predicted-vs-real, net of a (possibly brutal) fee
# ===========================================================================
@dataclass
class BacktestResult:
    times: np.ndarray
    lam: np.ndarray                 # the actual eigenvalue tape
    threshold: float
    fee: float
    pred_mean: np.ndarray           # OU one-step forecast at each t (NaN during warmup)
    pred_upper: np.ndarray
    predicted_idx: np.ndarray       # indices where OU predicted an arb (the red x)
    real_idx: np.ndarray            # indices where a fee-clearing loop actually existed
    # confusion of "predicted arb next tick" vs "arb actually there next tick"
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    net_edge: float = 0.0           # summed per-hop edge captured by firing on red x
    fired: int = 0                  # trades taken (= predicted arbs that had a next tick)
    wins: int = 0                   # of those, how many actually cleared the fee
    # --- naive PERSISTENCE baseline, scored on the SAME ticks: "predict arb next
    #     tick iff lambda clears tau NOW" (no model at all). OU only earns its keep
    #     if it beats these numbers -- otherwise its "forecast" is just autocorrelation.
    base_tp: int = 0
    base_fp: int = 0
    base_fn: int = 0
    base_tn: int = 0
    base_net_edge: float = 0.0
    base_fired: int = 0
    base_wins: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def hit_rate(self) -> float:
        return self.wins / self.fired if self.fired else float("nan")

    @property
    def base_precision(self) -> float:
        d = self.base_tp + self.base_fp
        return self.base_tp / d if d else float("nan")

    @property
    def base_recall(self) -> float:
        d = self.base_tp + self.base_fn
        return self.base_tp / d if d else float("nan")

    @property
    def base_hit_rate(self) -> float:
        return self.base_wins / self.base_fired if self.base_fired else float("nan")

    @property
    def beats_baseline(self) -> bool:
        """OU is worth its complexity only if it captures more net edge than persistence."""
        return self.net_edge > self.base_net_edge


def backtest(
    times: np.ndarray,
    lam: np.ndarray,
    fee: float = 0.002,
    window: int = 60,
    min_fit: int = 15,
    dt: float = 1.0,
    z: float = 1.96,
) -> BacktestResult:
    """
    Walk the tape left to right. At each t (once warm), fit OU on the trailing
    window, forecast the NEXT eigenvalue, and if the forecast clears tau(fee) call it
    a predicted arb (red x). Then look at what actually happened at t+1:

      * confusion matrix: predicted-arb-next vs actual-arb-next,
      * captured edge: if we fired (predicted arb), we "take" the loop and realise
        (lambda_{t+1} - tau) per hop -- POSITIVE if the loop was still open when we
        got there, NEGATIVE if it had already reverted (we paid the fee for nothing).
        Summed, this is the money mean-reversion leaves on the table once the fee is
        high enough that reversion beats execution.
    """
    n = len(lam)
    tau = fee_threshold(fee)
    pred = RollingArbitragePredictor(window=window, dt=dt, fee=fee, min_fit=min_fit, z=z)

    pred_mean = np.full(n, np.nan)
    pred_upper = np.full(n, np.nan)
    predicted_flags = np.zeros(n, dtype=bool)
    real_flags = lam > tau

    tp = fp = fn = tn = fired = wins = 0
    net_edge = 0.0
    b_tp = b_fp = b_fn = b_tn = b_fired = b_wins = 0
    b_net_edge = 0.0
    for i in range(n):
        p = pred.push(float(times[i]), float(lam[i]))
        if p is None:
            continue
        pred_mean[i] = p.mean_next
        pred_upper[i] = p.upper_next
        predicted_flags[i] = p.predicted_arb
        if i + 1 >= n:
            continue                                 # no next tick to score against
        actual_next = real_flags[i + 1]
        if p.predicted_arb and actual_next:
            tp += 1
        elif p.predicted_arb and not actual_next:
            fp += 1
        elif not p.predicted_arb and actual_next:
            fn += 1
        else:
            tn += 1
        if p.predicted_arb:                          # we would have fired
            fired += 1
            edge = float(lam[i + 1]) - tau           # realised per-hop edge next tick
            net_edge += edge
            if edge > 0:
                wins += 1

        # PERSISTENCE baseline on the SAME tick: no OU, just "arb now => arb next".
        base_arb = bool(real_flags[i])
        if base_arb and actual_next:
            b_tp += 1
        elif base_arb and not actual_next:
            b_fp += 1
        elif not base_arb and actual_next:
            b_fn += 1
        else:
            b_tn += 1
        if base_arb:
            b_fired += 1
            b_edge = float(lam[i + 1]) - tau
            b_net_edge += b_edge
            if b_edge > 0:
                b_wins += 1

    return BacktestResult(
        times=times,
        lam=lam,
        threshold=tau,
        fee=fee,
        pred_mean=pred_mean,
        pred_upper=pred_upper,
        predicted_idx=np.where(predicted_flags)[0],
        real_idx=np.where(real_flags)[0],
        tp=tp, fp=fp, fn=fn, tn=tn,
        net_edge=net_edge, fired=fired, wins=wins,
        base_tp=b_tp, base_fp=b_fp, base_fn=b_fn, base_tn=b_tn,
        base_net_edge=b_net_edge, base_fired=b_fired, base_wins=b_wins,
    )


def print_backtest_report(res: BacktestResult) -> None:
    def row(label: str, ou: str, base: str) -> None:
        print(f"  {label:<22}: {ou:>13}  {base:>13}")
    print("=" * 62)
    print(f"OU ARBITRAGE BACKTEST   (fee {res.fee*100:.2f}% / hop,  "
          f"tau = lambda>{res.threshold:+.6f})")
    print("-" * 62)
    print(f"  ticks scored          : {len(res.lam)}")
    print(f"  real arb ticks        : {len(res.real_idx)}")
    row("", "OU model", "persistence")
    row("confusion TP/FP/FN/TN",
        f"{res.tp}/{res.fp}/{res.fn}/{res.tn}",
        f"{res.base_tp}/{res.base_fp}/{res.base_fn}/{res.base_tn}")
    row("precision", f"{res.precision:.3f}", f"{res.base_precision:.3f}")
    row("recall", f"{res.recall:.3f}", f"{res.base_recall:.3f}")
    row("trades fired", f"{res.fired}", f"{res.base_fired}")
    row("win rate (edge>0)", f"{res.hit_rate:.3f}", f"{res.base_hit_rate:.3f}")
    row("net per-hop edge", f"{res.net_edge*100:+.4f}%", f"{res.base_net_edge*100:+.4f}%")
    print("-" * 62)
    # The whole point of the baseline: OU only earns its complexity if it captures
    # MORE net edge than the trivial "arb now => arb next" rule.
    delta = (res.net_edge - res.base_net_edge) * 100
    verdict = "OU BEATS persistence" if res.beats_baseline else "OU does NOT beat persistence"
    print(f"  verdict: {verdict}  ({delta:+.4f}%/hop net edge vs baseline)")
    print(f"           OU={'PROFIT' if res.net_edge > 0 else 'LOSS'}  "
          f"persistence={'PROFIT' if res.base_net_edge > 0 else 'LOSS'}   "
          f"(LOSS => reversion/fee ate it)")
    print("=" * 62)


# ===========================================================================
# 5. Visualisation -- rolling GIF + static summary PNG
# ===========================================================================
def _colors():
    return dict(actual="#1f77b4", forecast="#ff7f0e", band="#ffcc99",
                thresh="#555555", arb="#d62728", real="#2ca02c")


def render_summary_png(res: BacktestResult, path: str = "ou_backtest.png") -> str:
    """Static two-panel figure of the whole tape: series + red-x arbs, and the fan."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = _colors()
    PCT = 100.0                                  # per-hop log-return -> % per hop
    dt = float(res.times[1] - res.times[0]) if len(res.times) > 1 else 1.0
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

    ax.plot(res.times, res.lam * PCT, color=c["actual"], lw=1.2,
            label="tropical eigenvalue lambda(t)")
    # 1-step forecast made at t is FOR t+1, so plot it at t+dt: on a flat stretch it
    # sits on the actual (trivially right), but at each jump its one-tick lag shows.
    ax.plot(res.times + dt, res.pred_mean * PCT, color=c["forecast"], lw=1.0, ls="--",
            label="OU 1-step forecast (for t+1)")
    ax.fill_between(res.times + dt, res.pred_mean * PCT, res.pred_upper * PCT,
                    color=c["band"], alpha=0.4, label="OU 95% upper band")
    ax.axhline(res.threshold * PCT, color=c["thresh"], ls=":", lw=1.2,
               label=f"fee threshold tau ({res.fee*100:.3f}%/hop)")
    ax.axhline(0.0, color="#bbbbbb", lw=0.8)
    # real arbs (green o on the actual) and predicted arbs (red x on the forecast, at t+dt)
    if len(res.real_idx):
        ax.scatter(res.times[res.real_idx], res.lam[res.real_idx] * PCT, marker="o",
                   facecolors="none", edgecolors=c["real"], s=40, lw=1.2,
                   label="real arb (lambda>tau)")
    if len(res.predicted_idx):
        ax.scatter(res.times[res.predicted_idx] + dt, res.pred_mean[res.predicted_idx] * PCT,
                   marker="x", color=c["arb"], s=55, lw=1.8,
                   label="predicted arb (red x)")
    ax.set_ylabel("per-hop return  (%)")
    ax.set_title("OU model on the tropical eigenvalue -- arbitrage prediction vs reality")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.2)

    # bottom panel: the running "did the fired trade capture edge?" ledger
    tau = res.threshold
    fired_mask = np.zeros(len(res.lam), dtype=bool)
    fired_mask[res.predicted_idx] = True
    edge = np.where(fired_mask[:-1], res.lam[1:] - tau, 0.0)
    cum = np.concatenate([[0.0], np.cumsum(edge)]) * PCT
    ax2.plot(res.times, cum, color="#333333", lw=1.2)
    ax2.axhline(0.0, color="#bbbbbb", lw=0.8)
    ax2.fill_between(res.times, 0, cum, where=(cum >= 0), color=c["real"], alpha=0.3)
    ax2.fill_between(res.times, 0, cum, where=(cum < 0), color=c["arb"], alpha=0.3)
    ax2.set_ylabel("cum. edge (%)")
    ax2.set_xlabel("time")
    ax2.set_title(f"Cumulative per-hop edge from firing on every red x "
                  f"(net {res.net_edge*PCT:+.4f}%)", fontsize=9)
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return os.path.abspath(path)


def render_rolling_gif(
    times: np.ndarray,
    lam: np.ndarray,
    fee: float = 0.002,
    window: int = 60,
    min_fit: int = 15,
    horizon: int = 20,
    dt: float = 1.0,
    path: str = "ou_rolling.gif",
    fps: int = 12,
    max_frames: int = 240,
) -> str:
    """
    The scrolling window the user asked for: a FuncAnimation where the view holds the
    last `window` points, old data slides off the left, new data arrives on the
    right, and at each frame we overlay the OU forecast fan `horizon` steps ahead with
    a red x wherever the forecast clears tau(fee).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    c = _colors()
    tau = fee_threshold(fee)
    n = len(lam)
    start = min_fit
    frame_idx = list(range(start, n))
    if len(frame_idx) > max_frames:                  # subsample so the GIF stays small
        step = math.ceil(len(frame_idx) / max_frames)
        frame_idx = frame_idx[::step]

    PCT = 100.0                                      # per-hop log-return -> % per hop
    fig, ax = plt.subplots(figsize=(11, 5))
    ylo = (float(np.nanmin(lam)) - 0.002) * PCT
    yhi = (float(np.nanmax(lam)) + 0.004) * PCT

    def draw(i: int):
        ax.clear()
        lo = max(0, i - window + 1)
        tw, xw = times[lo:i + 1], lam[lo:i + 1]

        params = OUProcess.fit(xw, dt=dt)
        mean, lower, upper = OUProcess.forecast(params, float(xw[-1]), horizon=horizon)
        ft = times[i] + np.arange(1, horizon + 1) * dt

        ax.plot(tw, xw * PCT, color=c["actual"], lw=1.4, label="lambda(t) (observed)")
        ax.plot(ft, mean * PCT, color=c["forecast"], lw=1.4, ls="--", label="OU forecast")
        ax.fill_between(ft, lower * PCT, upper * PCT, color=c["band"], alpha=0.45, label="95% band")
        ax.axhline(tau * PCT, color=c["thresh"], ls=":", lw=1.2, label=f"tau (fee {fee*100:.3f}%)")
        ax.axhline(0.0, color="#bbbbbb", lw=0.8)

        # red x on the forecast wherever OU expects a fee-clearing loop
        arb = mean > tau
        if arb.any():
            ax.scatter(ft[arb], mean[arb] * PCT, marker="x", color=c["arb"], s=60, lw=2.0,
                       label="predicted arb", zorder=5)
        # green ring on observed points that were themselves real arbs
        real = xw > tau
        if real.any():
            ax.scatter(tw[real], xw[real] * PCT, marker="o", facecolors="none",
                       edgecolors=c["real"], s=45, lw=1.3, zorder=4)

        ax.set_xlim(tw[0], ft[-1])
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel("time")
        ax.set_ylabel("per-hop return  (%)")
        hl = params.half_life
        hl_s = f"{hl:.1f}" if math.isfinite(hl) else "inf"
        ax.set_title(f"OU rolling window  |  theta={params.theta:.3f}  "
                     f"mu={params.mu*PCT:+.4f}%  half-life={hl_s}")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(alpha=0.2)

    anim = FuncAnimation(fig, draw, frames=frame_idx, interval=1000 / fps)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return os.path.abspath(path)


# ===========================================================================
# 6. Demo / smoke test
# ===========================================================================
if __name__ == "__main__":
    # 1) synthesise a realistic eigenvalue tape (loops open then get closed).
    dt = 1.0
    t, lam = simulate_eigenvalue_series(n=400, dt=dt, seed=7)

    # 2) show the OU story at a MILD fee and at a BRUTAL fee -- reversion should eat
    #    the edge as the fee climbs (the user's "high fee / external friction" test).
    for fee in (0.001, 0.005):
        res = backtest(t, lam, fee=fee, window=60, min_fit=15, dt=dt)
        print_backtest_report(res)

    # 3) render the deliverables against the mild fee.
    res = backtest(t, lam, fee=0.001, window=60, min_fit=15, dt=dt)
    png = render_summary_png(res, path=os.path.join(_ROOT, "ou_backtest.png"))
    gif = render_rolling_gif(t, lam, fee=0.001, window=60, min_fit=15, horizon=20,
                             dt=dt, path=os.path.join(_ROOT, "ou_rolling.gif"))
    print(f"\nsummary PNG : {png}")
    print(f"rolling GIF : {gif}")

    # 4) prove the live hook: build one graph and read its eigenvalue through L2.
    assets = ["btc", "eth", "xrp", "sol"]
    snapshot = {
        "ethbtc": {"Binance": {"bid": "0.0610", "ask": "0.0611"},
                   "Kraken":  {"bid": "0.0600", "ask": "0.0601"}},
        "xrpbtc": {"Binance": {"bid": "0.00001200", "ask": "0.00001201"}},
        "xrpeth": {"Binance": {"bid": "0.00019600", "ask": "0.00019650"}},
    }
    g = ExchangeRateGraph(assets, transfer_cost=0.0).build_from_snapshot(snapshot)
    print(f"\nlive hook check: eigenvalue_from_graph(snapshot) = {eigenvalue_from_graph(g):+.6f}")
