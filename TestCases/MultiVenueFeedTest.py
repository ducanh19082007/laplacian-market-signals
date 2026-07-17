"""
Unit tests for L1_DataProcessing/MultiVenueFeed.py  (the aggregation layer).

MultiVenueFeed is the part that (a) fans a snapshot out over many venues,
(b) aggregates best prices, (c) reshapes into the arbitrage graph, and
(d) de-duplicates cycles for the live stream. The live parts (build_default_feed,
run_live, stream_arbitrage) spin up real WebSocket THREADS on construction, so the
tests here never touch the network:

  * pure logic (_cycle_signature)                    -> called directly, no object.
  * snapshot / aggregate_best_prices / get_all_pairs -> built on an EMPTY broker list
                                                        (no threads) with fake
                                                        dashboards injected by hand.
  * build_default_feed                               -> exercised with
                                                        OrderBookDashboard.run patched
                                                        to a no-op, so the 6 venue
                                                        configs get built and wired
                                                        but no socket ever opens.

Run:  pytest TestCases/MultiVenueFeedTest.py -v
"""

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import L1_DataProcessing.MultiVenueFeed as MVF
from L1_DataProcessing.MultiVenueFeed import (
    MultiBrokerOrderBook,
    BrokerConfig,
    build_default_feed,
    DEFAULT_FEE_TABLE,
)


# --------------------------------------------------------------------------- #
# a fake OrderBookDashboard so we can drive snapshot()/aggregate() offline     #
class _FakeDashboard:
    """
    Minimal stand-in exposing only the three methods MultiBrokerOrderBook calls:
    get_top_of_book, get_best_prices, get_quote_ts. Books are canned per pair.
    """

    def __init__(self, top_of_book, ts=None):
        # top_of_book: {pair: (bid, bid_size, ask, ask_size)}
        self._tob = top_of_book
        self._ts = ts or {}

    def get_top_of_book(self, pair):
        return self._tob.get(pair, ("N/A", "N/A", "N/A", "N/A"))

    def get_best_prices(self, pair):
        bid, _bs, ask, _as = self.get_top_of_book(pair)
        return bid, ask

    def get_quote_ts(self, pair):
        return self._ts.get(pair)


def _mbob_with_fakes(dashboards):
    """Empty broker list => no threads/network; then inject (BrokerConfig, fake)."""
    mbob = MultiBrokerOrderBook([], assets=["btc", "eth"])
    mbob.dashboards = dashboards
    return mbob


class TestCycleSignature:
    A, B, C = ("btc", "X"), ("eth", "X"), ("sol", "X")   # sorted: A < B < C

    def test_rotations_of_same_cycle_match(self):
        sig = MultiBrokerOrderBook._cycle_signature
        # closed cycle, plus two rotations starting at a different node.
        s1 = sig([self.A, self.B, self.C, self.A])
        s2 = sig([self.B, self.C, self.A, self.B])
        s3 = sig([self.C, self.A, self.B, self.C])
        assert s1 == s2 == s3
        # canonical form starts at the smallest node.
        assert s1 == (self.A, self.B, self.C)

    def test_open_ring_normalizes_too(self):
        sig = MultiBrokerOrderBook._cycle_signature
        assert sig([self.B, self.C, self.A]) == (self.A, self.B, self.C)

    def test_different_cycles_differ(self):
        sig = MultiBrokerOrderBook._cycle_signature
        D = ("xrp", "X")
        assert sig([self.A, self.B, self.C, self.A]) != sig([self.A, self.B, D, self.A])

    def test_empty_and_single(self):
        sig = MultiBrokerOrderBook._cycle_signature
        assert sig([]) == ()
        assert sig([self.A]) == (self.A,)


# --------------------------------------------------------------------------- #
# get_all_pairs / snapshot / aggregate_best_prices                            #
# --------------------------------------------------------------------------- #
class TestSnapshotAndAggregate:
    def test_get_all_pairs_is_sorted_union_lowercased(self):
        cfg1 = BrokerConfig(name="A", stream_url="mock:a", pairs=["ETHBTC", "xrpbtc"])
        cfg2 = BrokerConfig(name="B", stream_url="mock:b", pairs=["ethbtc", "solbtc"])
        mbob = _mbob_with_fakes([(cfg1, _FakeDashboard({})), (cfg2, _FakeDashboard({}))])
        assert mbob.get_all_pairs() == ["ethbtc", "solbtc", "xrpbtc"]

    def test_snapshot_shape_and_values(self):
        cfg = BrokerConfig(name="Binance", stream_url="mock:x", pairs=["ethbtc"])
        fake = _FakeDashboard(
            top_of_book={"ethbtc": (0.06, 1.0, 0.061, 2.0)},
            ts={"ethbtc": 1000.0},
        )
        mbob = _mbob_with_fakes([(cfg, fake)])
        snap = mbob.snapshot()
        q = snap["ethbtc"]["Binance"]
        assert q == {"bid": 0.06, "ask": 0.061, "bid_size": 1.0, "ask_size": 2.0, "ts": 1000.0}

    def test_aggregate_best_prices_picks_max_bid_min_ask(self):
        cfg1 = BrokerConfig(name="A", stream_url="mock:a", pairs=["ethbtc"])
        cfg2 = BrokerConfig(name="B", stream_url="mock:b", pairs=["ethbtc"])
        f1 = _FakeDashboard({"ethbtc": (0.060, 1, 0.061, 1)})
        f2 = _FakeDashboard({"ethbtc": (0.059, 1, 0.0605, 1)})
        mbob = _mbob_with_fakes([(cfg1, f1), (cfg2, f2)])
        agg = mbob.aggregate_best_prices("ethbtc")
        assert agg["best_bid"] == pytest.approx(0.060)   # highest bid wins
        assert agg["best_ask"] == pytest.approx(0.0605)  # lowest ask wins

    def test_aggregate_best_prices_all_na(self):
        cfg = BrokerConfig(name="A", stream_url="mock:a", pairs=["ethbtc"])
        # no book for the requested pair -> fake returns N/A -> filtered out.
        mbob = _mbob_with_fakes([(cfg, _FakeDashboard({}))])
        agg = mbob.aggregate_best_prices("ethbtc")
        assert agg["best_bid"] == "N/A"
        assert agg["best_ask"] == "N/A"

    def test_build_graph_none_without_assets(self):
        mbob = MultiBrokerOrderBook([])          # assets=None
        assert mbob.build_graph() is None

    def test_build_graph_reshapes_snapshot(self):
        cfg = BrokerConfig(name="Binance", stream_url="mock:x", pairs=["ethbtc"])
        fake = _FakeDashboard(
            top_of_book={"ethbtc": (0.06, 5.0, 0.061, 5.0)},
            ts={"ethbtc": 1000.0},
        )
        mbob = _mbob_with_fakes([(cfg, fake)])
        mbob.assets = ["btc", "eth"]
        graph = mbob.build_graph()
        assert graph is not None
        # convert edges exist and weights are populated (log_transform ran).
        eth, btc = ("eth", "Binance"), ("btc", "Binance")
        assert graph.adjacency[eth][btc]["rate"] == pytest.approx(0.06)
        assert graph.adjacency[eth][btc]["weight"] is not None


# --------------------------------------------------------------------------- #
# build_default_feed -- wired offline via a no-op run()                        #
# --------------------------------------------------------------------------- #
class TestBuildDefaultFeed:
    @pytest.fixture(autouse=True)
    def _no_network(self, monkeypatch):
        # OrderBookDashboard.run() is what spawns the websocket thread; stub it so
        # build_default_feed constructs the whole 6-venue universe but stays offline.
        monkeypatch.setattr(MVF.OrderBookDashboard, "run", lambda self: None)

    def test_six_venues_in_order(self):
        feed = build_default_feed()
        names = [cfg.name for cfg, _ in feed.dashboards]
        assert names == ["Binance", "Coinbase Adv.", "Kraken", "OKX", "Gemini", "Bitstamp"]

    def test_defaults_wired_through(self):
        feed = build_default_feed()
        assert feed.fee == DEFAULT_FEE_TABLE            # realistic per-venue table
        assert feed.quote_window == 0.2
        assert feed.min_notional["usd"] == 50.0
        assert feed.min_notional["btc"] == 0.0005
        assert feed.min_notional["eth"] == 0.02
        assert {"btc", "eth", "usdt", "usdc"} <= set(feed.assets)

    def test_globally_unsupported_pairs_excluded(self):
        feed = build_default_feed()
        pairs = feed.get_all_pairs()
        assert "solxrp" not in pairs and "xrpsol" not in pairs

    def test_every_generated_pair_has_a_quote_side(self):
        # make_pair should never emit an alt-vs-alt pair (no stable/BTC/ETH quote).
        feed = build_default_feed()
        quotes = ("usdt", "usdc", "btc", "eth", "usd", "eur", "gbp")
        assert all(any(p.endswith(q) for q in quotes) for p in feed.get_all_pairs())

    def test_overrides_scalar_fee_and_disabled_notional(self):
        feed = build_default_feed(fee=0.002, min_notional=None)
        assert feed.fee == 0.002
        assert feed.min_notional is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
