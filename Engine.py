"""
Engine.py -- centralized orchestrator for the FOREX_farming pipeline.

This is the OUTER loop. Each layer (L1..L5) is a thin component the engine
calls once per tick; the engine owns config, the tick loop, the gate logic,
and rendering. That separation is the whole point:

    - L1 only produces a graph (it does NOT own a `while True` anymore).
    - L2/L3/L5 are stubbed today (Null*) and swapped in later WITHOUT
      touching this loop.
    - L4 already works -- BellmanFordDetector just wraps the existing
      ExchangeRateGraph.find_arbitrage().

Run it:   python Engine.py        (from the repo root)

Author: Anh Duc Le
"""

import time
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Protocol

from L1_DataProcessing.MultiVenueFeed import (
    MultiBrokerOrderBook,
    BrokerConfig,
    URL_methods,
)
from L1_DataProcessing.DataProcessing import ExchangeRateGraph, Node


def _now() -> str:
    return time.strftime("%H:%M:%S")


# ===========================================================================
# Per-layer result types (what flows between stages)
# ===========================================================================
@dataclass
class Structure:
    """L2 output. tropical_bound = +inf means 'no usable bound' (never gate)."""
    spectral_gap: Optional[float] = None       # λ₂, Fiedler value
    tropical_bound: float = float("inf")        # upper bound on best cycle return
    strain: Optional[float] = None              # deviation from no-arb equilibrium


@dataclass
class Cycle:
    """L4 output: one arbitrage loop in the graph."""
    path: List[Node]
    ret: float                                  # product of rates (>1 = profit)

    @property
    def profit_pct(self) -> float:
        return (self.ret - 1.0) * 100.0

    def signature(self) -> tuple:
        return tuple(self.path)

    def render(self) -> str:
        path = " -> ".join(ExchangeRateGraph.fmt(n) for n in self.path)
        return f"[{_now()}] {self.profit_pct:+.4f}%  ({self.ret:.8f})  {path}"


@dataclass
class TickResult:
    """Everything one pass through the pipeline produced."""
    timestamp: str
    structure: Optional[Structure] = None
    cycles: List[Cycle] = field(default_factory=list)
    skipped_reason: Optional[str] = None        # set when the gate short-circuits


# ===========================================================================
# Layer interfaces -- implement these per layer; swap into MarketEngine later
# ===========================================================================
class StructureAnalyzer(Protocol):              # L2: Laplacian, λ₂, tropical, strain
    def analyze(self, g: ExchangeRateGraph) -> Structure: ...


class SpatialAnalyzer(Protocol):                # L3: Tarjan SCC -> tradeable subgraphs
    def tradeable(self, g: ExchangeRateGraph) -> Optional[List[set]]: ...


class RegimeEngine(Protocol):                   # L5: OU calibration, ADF, Monte Carlo
    def update(self, strain: Optional[float]) -> None: ...
    def is_mean_reverting(self, cycle: List[Node]) -> bool: ...


class ArbitrageDetector(Protocol):              # L4: Bellman-Ford + cost filter
    def find(self, g: ExchangeRateGraph,
             sccs: Optional[List[set]],
             regime: RegimeEngine) -> List[Cycle]: ...


# ===========================================================================
# Stub implementations -- let the engine RUN today on L1 + L4 only
# ===========================================================================
class NullStructure:
    """No structural analysis yet: returns a bound that never gates detection."""
    def analyze(self, g: ExchangeRateGraph) -> Structure:
        return Structure()                      # tropical_bound = +inf


class NullSpatial:
    """No SCC pruning yet: None means 'search the whole graph'."""
    def tradeable(self, g: ExchangeRateGraph) -> Optional[List[set]]:
        return None


class NullRegime:
    """No regime gate yet: every cycle is trusted."""
    def update(self, strain: Optional[float]) -> None:
        pass

    def is_mean_reverting(self, cycle: List[Node]) -> bool:
        return True


class BellmanFordDetector:
    """L4, real today -- wraps ExchangeRateGraph.find_arbitrage()."""
    def find(self, g, sccs, regime) -> List[Cycle]:
        # sccs is ignored until L3 lands; Bellman-Ford runs over the whole graph.
        cycle = g.find_arbitrage()
        if not cycle:
            return []
        if not regime.is_mean_reverting(cycle):  # L5 gate (no-op under NullRegime)
            return []
        return [Cycle(path=cycle, ret=g.cycle_return(cycle))]


# ===========================================================================
# The orchestrator
# ===========================================================================
@dataclass
class EngineConfig:
    assets: List[str]
    brokers: List[BrokerConfig]
    refresh_interval: float = 0.05
    transfer_cost: float = 0.0
    fee: float = 0.0020                         # taker fee per convert leg (0.20%)
    max_quote_age: float = 1.0                  # absolute backstop: drop quotes older than 1s
    quote_window: float = 0.5                   # legs of a cycle must be within 0.5s (kills stale-leg phantoms)
    # Min tradeable top-of-book notional per quote currency; edges thinner than this
    # are dropped (kills phantom arb off illiquid/mispriced books). None disables it.
    min_notional: Optional[Dict[str, float]] = None
    min_profit: float = 0.0005                  # drop cycles under 0.05% net return
    cost_threshold: float = 0.0                 # tropical early-exit gate
    history: int = 20                           # detections kept on screen
    only_on_change: bool = True                 # log a cycle only when it changes


class MarketEngine:
    def __init__(
        self,
        cfg: EngineConfig,
        structure: Optional[StructureAnalyzer] = None,
        spatial: Optional[SpatialAnalyzer] = None,
        regime: Optional[RegimeEngine] = None,
        detector: Optional[ArbitrageDetector] = None,
    ):
        self.cfg = cfg
        self.feed = MultiBrokerOrderBook(
            cfg.brokers,
            refresh_interval=cfg.refresh_interval,
            assets=cfg.assets,
            transfer_cost=cfg.transfer_cost,
            fee=cfg.fee,
            max_quote_age=cfg.max_quote_age,
            quote_window=cfg.quote_window,
            min_notional=cfg.min_notional,
        )
        # default = stubs, so the engine runs on L1 + L4 out of the box.
        self.structure = structure or NullStructure()
        self.spatial = spatial or NullSpatial()
        self.regime = regime or NullRegime()
        self.detector = detector or BellmanFordDetector()

        # render state (the only_on_change dedup + sliding detection window)
        self._log: deque = deque(maxlen=cfg.history)
        self._last_signature: Optional[tuple] = None

    # ----------------------------------------------------------------- one pass
    def tick(self) -> TickResult:
        """One pass through stages 1-5. No loop here -- run() drives it."""
        g = self.feed.build_graph()                              # L1
        if g is None:
            return TickResult(_now())

        s = self.structure.analyze(g)                            # L2
        self.regime.update(s.strain)                             # feed L5 the strain

        # Gate: if the best possible cycle can't beat costs, skip detection.
        if s.tropical_bound < self.cfg.cost_threshold:
            return TickResult(_now(), s, skipped_reason="tropical bound below cost")

        sccs = self.spatial.tradeable(g)                         # L3
        cycles = self.detector.find(g, sccs, self.regime)        # L4 (regime-gated)
        # Drop cycles that don't clear min_profit -- after fees, anything under a
        # few bps is float noise, not a tradeable edge.
        cycles = [c for c in cycles if (c.ret - 1.0) > self.cfg.min_profit]
        return TickResult(_now(), s, cycles)

    # --------------------------------------------------------------- outer loop
    def run(self) -> None:
        """The live tick loop -- the OUTER loop lives here, not in L1."""
        try:
            while True:
                result = self.tick()
                self._note(result)
                self._render(result)
                time.sleep(self.cfg.refresh_interval)
        except KeyboardInterrupt:
            self._shutdown()
            print("\nEngine stopped.")

    # --------------------------------------------------------------- rendering
    def _note(self, result: TickResult) -> None:
        """Roll the newest cycle into the sliding detection window."""
        if not result.cycles:
            self._last_signature = None
            return
        top = result.cycles[0]
        sig = top.signature()
        if not self.cfg.only_on_change or sig != self._last_signature:
            self._log.append(top.render())
            self._last_signature = sig

    def _render(self, result: TickResult) -> None:
        # Reuse L1's tested box primitives; the engine owns the loop + detections.
        print("\033[H\033[J", end="")
        right = ["ARBITRAGE DETECTIONS", "-" * 20]
        if result.skipped_reason:
            right.append(f"(skipped: {result.skipped_reason})")
        right += list(self._log) if self._log else ["(none yet)"]
        print(self.feed._render_side_by_side(self.feed._exchange_rate_box(), right))

    def _shutdown(self) -> None:
        for _, dashboard in self.feed.dashboards:
            dashboard.is_running = False


# ===========================================================================
# Default wiring -- the "bigger" 7-asset multi-venue graph across 3 venues
# ===========================================================================
def default_config() -> EngineConfig:
    assets = ["btc", "eth", "xrp", "sol", "ada", "usdc", "doge"]
    quote_priority = assets

    def make_pair(a: str, b: str) -> str:
        base, quote = (b, a) if quote_priority.index(a) < quote_priority.index(b) else (a, b)
        return f"{base}{quote}"

    # SOL/XRP has no native book on any of the three venues -- drop it.
    GLOBALLY_UNSUPPORTED = {"solxrp", "xrpsol"}
    my_pairs = [
        make_pair(a, b)
        for a, b in combinations(assets, 2)
        if make_pair(a, b) not in GLOBALLY_UNSUPPORTED
    ]
    # Coinbase quotes XRP only vs USD/USDC -- keep rows for the table, skip the sub.
    COINBASE_UNSUPPORTED = {"xrpbtc", "xrpeth"}
    coinbase_pairs = [p for p in my_pairs if p not in COINBASE_UNSUPPORTED]

    brokers = [
        BrokerConfig(
            name="Binance",
            stream_url=URL_methods.make_binance_depth_url(my_pairs),
            pairs=my_pairs,
        ),
        BrokerConfig(
            name="Coinbase Adv.",
            stream_url="wss://advanced-trade-ws.coinbase.com",
            pairs=my_pairs,
            payload_extractor=URL_methods.make_coinbase_payload_extractor(),
            initial_message=URL_methods.make_coinbase_subscription_message(coinbase_pairs, assets),
        ),
        BrokerConfig(
            name="Kraken",
            stream_url="wss://ws.kraken.com/v2",
            pairs=my_pairs,
            payload_extractor=URL_methods.make_kraken_payload_extractor(),
            initial_message=URL_methods.make_kraken_subscription_message(my_pairs, assets),
        ),
    ]
    # Drop edges whose top-of-book is too thin to actually trade (~$50 equivalent),
    # the main guard against phantom single-venue triangles on illiquid books.
    min_notional = {
        "usd": 50.0, "usdt": 50.0, "usdc": 50.0, "eur": 50.0, "gbp": 50.0,
        "btc": 0.0005, "eth": 0.02,
    }
    return EngineConfig(
        assets=assets, brokers=brokers, refresh_interval=0.05, min_notional=min_notional
    )


if __name__ == "__main__":
    # Runs today: L1 feed + L4 Bellman-Ford, L2/L3/L5 are no-op stubs.
    # Swap a real layer in by passing it to MarketEngine, e.g.:
    #     MarketEngine(cfg, structure=GraphLaplacianAnalyzer(), regime=OURegimeEngine())
    MarketEngine(default_config()).run()
