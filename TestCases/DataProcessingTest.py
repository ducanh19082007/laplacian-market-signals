"""
Unit tests for L1_DataProcessing/DataProcessing.py  (the ExchangeRateGraph layer).

Everything here is PURE, deterministic reshaping of a snapshot dict + a math.log,
so it needs no network, no threads and no L3 extension -- we build tiny hand-rigged
snapshots and assert the exact adjacency/rates/weights that come out.

Run just this file:      pytest TestCases/DataProcessingTest.py -v
Run the whole L1 suite:  pytest TestCases -v

Author of tests: (scaffolding) -- extend freely.
"""

import math
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from L1_DataProcessing.DataProcessing import (
    ExchangeRateGraph,
    split_pair,
    _to_float,
)


class TestSplitPair:
    def test_basic_split(self):
        assert split_pair("ethbtc", ["btc", "eth", "xrp"]) == ("eth", "btc")

    def test_case_insensitive(self):
        # both the pair and the asset list may arrive upper/mixed case
        assert split_pair("ETHBTC", ["BTC", "ETH"]) == ("eth", "btc")

    def test_longest_quote_wins_when_prefixed(self):
        # 'usdusdt' must resolve to base=usd, quote=usdt (endswith usdt),
        # NOT base='' quote='usd' -- the empty-base guard forces the good split.
        assert split_pair("usdusdt", ["usd", "usdt"]) == ("usd", "usdt")

    def test_unknown_base_returns_none(self):
        # 'doge' is not a known asset, so 'dogebtc' cannot be split.
        assert split_pair("dogebtc", ["btc", "eth"]) is None

    def test_no_base_returns_none(self):
        # pair IS the quote with nothing left over -> not a real pair.
        assert split_pair("btc", ["btc"]) is None

    def test_unrelated_string_returns_none(self):
        assert split_pair("hello", ["btc", "eth"]) is None


class TestToFloat:
    def test_valid_positive(self):
        assert _to_float("1.5") == 1.5
        assert _to_float(2) == 2.0

    def test_zero_is_none(self):
        # a price/size of 0 is not tradeable -> treated as missing.
        assert _to_float("0") is None
        assert _to_float(0) is None

    def test_negative_is_none(self):
        assert _to_float("-3") is None

    def test_garbage_is_none(self):
        assert _to_float("N/A") is None
        assert _to_float(None) is None
        assert _to_float([1, 2]) is None


class TestInit:
    def test_lowercases_and_defaults(self):
        g = ExchangeRateGraph(
            ["BTC", "ETH"],
            transfer_cost=0.1,
            fee=0.002,
            quote_window=0.5,
            min_notional={"USD": 50.0},
        )
        assert g.assets == ["btc", "eth"]
        assert g.min_notional == {"usd": 50.0}     # keys lowercased too
        assert g.transfer_cost == 0.1
        assert g.fee == 0.002
        assert g.quote_window == 0.5
        assert g.adjacency == {}


class TestFeeFor:
    """This is the method the original stub `test_feefor` was reaching for."""

    def test_scalar_fee_is_uniform(self):
        g = ExchangeRateGraph([], fee=0.001)
        assert g._fee_for("Binance") == 0.001
        assert g._fee_for("AnyVenue") == 0.001

    def test_dict_fee_per_broker(self):
        g = ExchangeRateGraph([], fee={"Binance": 0.001, "Kraken": 0.0026, "default": 0.005})
        assert g._fee_for("Binance") == 0.001
        assert g._fee_for("Kraken") == 0.0026

    def test_dict_fee_falls_back_to_default(self):
        g = ExchangeRateGraph([], fee={"Binance": 0.001, "default": 0.005})
        assert g._fee_for("SomeUnlistedVenue") == 0.005

    def test_dict_fee_no_default_is_zero(self):
        g = ExchangeRateGraph([], fee={"Binance": 0.001})
        assert g._fee_for("Unlisted") == 0.0

    def test_zero_scalar(self):
        assert ExchangeRateGraph([], fee=0.0)._fee_for("X") == 0.0


class TestAddEdge:
    def test_valid_edge_registers_both_nodes(self):
        g = ExchangeRateGraph(["btc", "eth"])
        a, b = ("eth", "X"), ("btc", "X")
        g._add_edge(a, b, 1.5, kind="convert")
        assert a in g.adjacency and b in g.adjacency          # both endpoints exist
        assert g.adjacency[a][b]["rate"] == 1.5
        assert g.adjacency[a][b]["weight"] is None            # filled by log_transform later
        assert g.adjacency[a][b]["kind"] == "convert"

    @pytest.mark.parametrize("bad_rate", [0, -1, -0.0001, None])
    def test_nonpositive_or_none_rate_rejected(self, bad_rate):
        g = ExchangeRateGraph(["btc", "eth"])
        g._add_edge(("eth", "X"), ("btc", "X"), bad_rate)
        assert g.adjacency == {}
        
    
    def test_add_edge_overwrites_existing_edge(self):
        g = ExchangeRateGraph(["btc", "eth"])
        a, b = ("eth", "X"), ("btc", "X")
        
        # First insertion
        g._add_edge(a, b, 1.5, kind="convert")
        # Overwrite insertion
        g._add_edge(a, b, 2.0, kind="direct")
        
        assert g.adjacency[a][b]["rate"] == 2.0
        assert g.adjacency[a][b]["kind"] == "direct"
    def test_self_loop_edge_handling(self):
        g = ExchangeRateGraph(["btc"])
        a = ("btc", "X")
        
        # Testing how the graph handles a node mapping to itself
        g._add_edge(a, a, 1.0, kind="identity")
        
        assert a in g.adjacency
        assert g.adjacency[a][a]["rate"] == 1.0
        
    
    def test_transfer_edges_creates_full_clique_minus_self(self):
        g = ExchangeRateGraph([])
        g.transfer_cost = 0.01
        
        # 4 brokers for 1 asset means 4 * 3 = 12 directed edges
        brokers_data = {"btc": {"binance", "coinbase", "kraken", "okx"}}
        g._add_transfer_edges(brokers_data)
        
        expected_rate = 1.0 - g.transfer_cost
        total_edges = 0
        
        for u in g.adjacency:
            for v in g.adjacency[u]:
                total_edges += 1
                assert u != v  # No self loops
                assert u[0] == v[0] == "btc"  # Asset remains identical
                assert g.adjacency[u][v]["rate"] == expected_rate
                assert g.adjacency[u][v]["rate_raw"] == 1.0
                assert g.adjacency[u][v]["fee"] == g.transfer_cost
                assert g.adjacency[u][v]["kind"] == "transfer"
                
        assert total_edges == 12

    def test_transfer_edges_remains_strictly_siloed_per_asset(self):
        g = ExchangeRateGraph([])
        g.transfer_cost = 0.002
        
        brokers_data = {
            "btc": {"binance", "coinbase"},
            "eth": {"coinbase", "kraken"}
        }
        
        g._add_transfer_edges(brokers_data)
        
        # Verify cross-asset edges do not exist
        # coinbase has both btc and eth, but they must never map to each other
        btc_coinbase = ("btc", "coinbase")
        eth_coinbase = ("eth", "coinbase")
        
        if btc_coinbase in g.adjacency:
            assert eth_coinbase not in g.adjacency[btc_coinbase]
            
        if eth_coinbase in g.adjacency:
            assert btc_coinbase not in g.adjacency[eth_coinbase]


class TestBuildFromSnapshot:
    def test_single_pair_two_directions_no_fee(self):
        g = ExchangeRateGraph(["btc", "eth"])
        g.build_from_snapshot({"ethbtc": {"Binance": {"bid": "0.06", "ask": "0.061"}}})

        eth, btc = ("eth", "Binance"), ("btc", "Binance")
        # Sell ETH -> BTC at the bid (quote per base).
        assert g.adjacency[eth][btc]["rate"] == pytest.approx(0.06)
        assert g.adjacency[eth][btc]["kind"] == "convert"
        assert g.adjacency[eth][btc]["pair"] == "ethbtc"
        assert g.adjacency[eth][btc]["broker"] == "Binance"
        assert g.adjacency[eth][btc]["rate_raw"] == pytest.approx(0.06)
        assert g.adjacency[eth][btc]["fee"] == 0.0
        # Buy ETH with BTC -> 1/ask base per quote.
        assert g.adjacency[btc][eth]["rate"] == pytest.approx(1.0 / 0.061)
        # exactly two convert edges, no transfer edges (single venue).
        assert len(g.edges()) == 2

    def test_fee_scalar_applied_to_net_rate_only(self):
        g = ExchangeRateGraph(["btc", "eth"], fee=0.01)
        g.build_from_snapshot({"ethbtc": {"Binance": {"bid": "0.06", "ask": "0.061"}}})
        edge = g.adjacency[("eth", "Binance")][("btc", "Binance")]
        assert edge["rate"] == pytest.approx(0.06 * 0.99)   # net-of-fee (executable)
        assert edge["rate_raw"] == pytest.approx(0.06)      # fee-free market rate kept
        assert edge["fee"] == 0.01

    def test_fee_dict_per_broker_applied(self):
        g = ExchangeRateGraph(["btc", "eth"], fee={"Binance": 0.001, "default": 0.005})
        g.build_from_snapshot({"ethbtc": {"Binance": {"bid": "0.06", "ask": "0.061"}}})
        edge = g.adjacency[("eth", "Binance")][("btc", "Binance")]
        assert edge["rate"] == pytest.approx(0.06 * (1 - 0.001))
        assert edge["fee"] == 0.001

    def test_crossed_book_is_dropped(self):
        # bid >= ask can only come from out-of-sync/rounded updates; left in it
        # fabricates a same-venue round-trip "arbitrage".
        g = ExchangeRateGraph(["btc", "eth"])
        g.build_from_snapshot({"ethbtc": {"Binance": {"bid": "0.061", "ask": "0.060"}}})
        assert g.adjacency == {}
        assert g.edges() == []

    def test_unknown_pair_skipped(self):
        g = ExchangeRateGraph(["btc", "eth"])
        g.build_from_snapshot({"dogexrp": {"Binance": {"bid": "1", "ask": "2"}}})
        assert g.adjacency == {}

    def test_transfer_edges_stitch_same_asset_across_venues(self):
        g = ExchangeRateGraph(["btc", "eth"], quote_window=0.1)
        snap = {
            "ethbtc": {
                "Binance": {"bid": "0.060", "ask": "0.061", "ts": 1000.0},
                "Kraken":  {"bid": "0.059", "ask": "0.0595", "ts": 1000.0},
            }
        }
        g.build_from_snapshot(snap)
        # same asset, different venue -> 1:1 transfer (transfer_cost defaults to 0).
        t = g.adjacency[("eth", "Binance")][("eth", "Kraken")]
        assert t["kind"] == "transfer"
        assert t["rate"] == pytest.approx(1.0)
        assert t["rate_raw"] == 1.0        # transfers are structurally fee-free
        assert ("btc", "Binance") in g.adjacency[("btc", "Kraken")]

    def test_transfer_cost_reduces_transfer_rate(self):
        g = ExchangeRateGraph(["btc", "eth"], transfer_cost=0.002, quote_window=0.1)
        snap = {
            "ethbtc": {
                "Binance": {"bid": "0.060", "ask": "0.061", "ts": 1000.0},
                "Kraken":  {"bid": "0.059", "ask": "0.0595", "ts": 1000.0},
            }
        }
        g.build_from_snapshot(snap)
        assert g.adjacency[("eth", "Binance")][("eth", "Kraken")]["rate"] == pytest.approx(0.998)

    def test_quote_window_drops_stale_leg(self):
        # Kraken lags the freshest quote by 1s but window is 0.1s -> its quotes
        # are treated as stale and dropped, leaving only Binance (no transfers).
        g = ExchangeRateGraph(["btc", "eth"], quote_window=0.1)
        snap = {
            "ethbtc": {
                "Binance": {"bid": "0.060", "ask": "0.061", "ts": 1000.0},
                "Kraken":  {"bid": "0.059", "ask": "0.0595", "ts": 999.0},
            }
        }
        g.build_from_snapshot(snap)
        assert ("eth", "Kraken") not in g.adjacency
        assert ("eth", "Binance") in g.adjacency

    def test_missing_ts_under_active_window_is_stale(self):
        g = ExchangeRateGraph(["btc", "eth"], quote_window=0.1)
        snap = {
            "ethbtc": {
                "Binance": {"bid": "0.060", "ask": "0.061", "ts": 1000.0},
                "Kraken":  {"bid": "0.059", "ask": "0.0595"},   # no ts -> stale
            }
        }
        g.build_from_snapshot(snap)
        assert ("eth", "Kraken") not in g.adjacency

    def test_min_notional_drops_thin_book(self):
        # min_notional on btc but no sizes supplied -> both sides untradeable.
        g = ExchangeRateGraph(["btc", "eth"], min_notional={"btc": 0.0005})
        g.build_from_snapshot({"ethbtc": {"Binance": {"bid": "0.06", "ask": "0.061"}}})
        assert g.adjacency == {}

    def test_min_notional_keeps_deep_book(self):
        g = ExchangeRateGraph(["btc", "eth"], min_notional={"btc": 0.0005})
        snap = {
            "ethbtc": {
                "Binance": {"bid": "0.06", "ask": "0.061", "bid_size": "1", "ask_size": "1"}
            }
        }
        g.build_from_snapshot(snap)
        # 0.06 * 1 = 0.06 >= 0.0005 -> both edges survive.
        assert len(g.edges()) == 2

    def test_rebuild_is_idempotent(self):
        # build_from_snapshot resets self.adjacency first, so calling twice on the
        # same snapshot must not double the edges.
        g = ExchangeRateGraph(["btc", "eth"])
        snap = {"ethbtc": {"Binance": {"bid": "0.06", "ask": "0.061"}}}
        g.build_from_snapshot(snap)
        g.build_from_snapshot(snap)
        assert len(g.edges()) == 2
        
        
    def test_snapshot_builds_correct_bid_ask_edges(self):
        assets = ["btc", "eth", "xrp", "sol"]
        snapshot = {
            "ethbtc": {
                "Binance": {"bid": "0.0610", "ask": "0.0611"},
                "Kraken":  {"bid": "0.0600", "ask": "0.0601"},
            },
            "xrpbtc": {
                "Binance": {"bid": "0.00001200", "ask": "0.00001201"},
            },
            "xrpeth": {
                "Binance": {"bid": "0.00019600", "ask": "0.00019650"},
            },
        }
        
        g = ExchangeRateGraph(assets)
        g.build_from_snapshot(snapshot)
        
        # Verify directional rates for eth <-> btc on Binance
        eth_binance = ("eth", "Binance")
        btc_binance = ("btc", "Binance")
        
        # Selling ETH for BTC (Bid) -> rate is 0.0610
        assert g.adjacency[eth_binance][btc_binance]["rate"] == 0.0610
        # Buying ETH with BTC (Ask) -> rate is 1 / 0.0611
        assert g.adjacency[btc_binance][eth_binance]["rate"] == pytest.approx(1.0 / 0.0611)

    def test_quote_window_drops_stale_timestamps(self):
        assets = ["btc", "eth", "xrp", "sol"]
        snapshot1 = {
            "ethbtc": {
                "Binance": {"bid": "0.0610", "ask": "0.0611", "ts": 1000.0},
                "Kraken": {"bid": "0.0600", "ask": "0.0601", "ts": 1000.676767},
            },
            "xrpbtc": {
                "Binance": {"bid": "0.00001200", "ask": "0.00001201", "ts": 1000.0},
            },
            "xrpeth": {
                "Binance": {"bid": "0.00019600", "ask": "0.00019650", "ts": 1000.0},
            },
        }

        # Max timestamp is 1000.676767 (Kraken). Window is 0.1.
        # Anything older than 1000.576767 (like Binance at 1000.0) must be discarded.
        g = ExchangeRateGraph(assets, transfer_cost=0.0, quote_window=0.1)
        g.build_from_snapshot(snapshot1)

        # Kraken should survive
        assert ("eth", "Kraken") in g.adjacency
        
        # Binance pairs should be completely pruned/absent due to staleness
        assert ("eth", "Binance") not in g.adjacency
        assert ("xrp", "Binance") not in g.adjacency

class TestLogTransform:
    def test_weight_is_negative_log_rate(self):
        g = ExchangeRateGraph(["btc", "eth"])
        a, b = ("eth", "X"), ("btc", "X")
        g._add_edge(a, b, 2.0)
        g._add_edge(b, a, 0.5)
        g.log_transform()
        assert g.adjacency[a][b]["weight"] == pytest.approx(-math.log(2.0))
        assert g.adjacency[b][a]["weight"] == pytest.approx(-math.log(0.5))  # positive

    def test_unit_rate_gives_zero_weight(self):
        g = ExchangeRateGraph(["btc"])
        g._add_edge(("btc", "A"), ("btc", "B"), 1.0)
        g.log_transform()
        assert g.adjacency[("btc", "A")][("btc", "B")]["weight"] == pytest.approx(0.0)

    def test_profitable_cycle_sums_to_negative_weight(self):
        # product(rate) > 1  <=>  sum(-ln rate) < 0  (a negative cycle).
        g = ExchangeRateGraph(["btc", "eth"])
        a, b = ("eth", "X"), ("btc", "X")
        g._add_edge(a, b, 0.06)
        g._add_edge(b, a, 1.0 / 0.059)     # rebuy cheaper -> round-trip profits
        g.log_transform()
        loop_weight = g.adjacency[a][b]["weight"] + g.adjacency[b][a]["weight"]
        assert loop_weight < 0

class TestGraphHelpers:
    def _triangle(self):
        g = ExchangeRateGraph(["btc", "eth", "sol"])
        self.A, self.B, self.C = ("btc", "X"), ("eth", "X"), ("sol", "X")
        g._add_edge(self.A, self.B, 2.0)
        g._add_edge(self.B, self.C, 3.0)
        g._add_edge(self.C, self.A, 4.0)
        return g

    def test_nodes_are_sorted(self):
        g = self._triangle()
        assert g.nodes() == sorted([self.A, self.B, self.C])

    def test_edges_count(self):
        g = self._triangle()
        assert len(g.edges()) == 3

    def test_cycle_return_multiplies_rates(self):
        g = ExchangeRateGraph(["btc", "eth"])
        a, b = ("eth", "X"), ("btc", "X")
        g._add_edge(a, b, 0.5)
        g._add_edge(b, a, 2.0)
        assert g.cycle_return([a, b, a]) == pytest.approx(1.0)

    def test_cycle_return_zero_on_broken_edge(self):
        g = self._triangle()
        # A -> C edge does not exist (only A->B, B->C, C->A).
        assert g.cycle_return([self.A, self.C]) == 0.0

    def test_subgraph_keeps_only_internal_edges(self):
        g = self._triangle()
        sub = g.subgraph([self.A, self.B])
        # only A->B has both endpoints inside {A,B}.
        assert len(sub.edges()) == 1
        assert sub.edges()[0][0] == self.A and sub.edges()[0][1] == self.B

    def test_subgraph_is_non_destructive(self):
        g = self._triangle()
        g.subgraph([self.A, self.B])
        assert len(g.edges()) == 3          # original untouched

    def test_subgraph_shares_edge_attr_dict_by_reference(self):
        # the L3->L1 reduction reuses the already-computed weight dicts, no rebuild.
        g = self._triangle()
        sub = g.subgraph([self.A, self.B])
        assert sub.adjacency[self.A][self.B] is g.adjacency[self.A][self.B]

    def test_fmt(self):
        assert ExchangeRateGraph.fmt(("eth", "Binance")) == "ETH@Binance"

    def test_summary_is_string_with_counts(self):
        # summary() renders each edge's "kind"/"pair" attrs, so it expects a graph
        # built the normal way (build_from_snapshot), not bare _add_edge stubs.
        g = ExchangeRateGraph(["btc", "eth"])
        g.build_from_snapshot({"ethbtc": {"Binance": {"bid": "0.06", "ask": "0.061"}}})
        g.log_transform()
        s = g.summary()
        assert isinstance(s, str)
        assert "nodes" in s and "edges" in s


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
