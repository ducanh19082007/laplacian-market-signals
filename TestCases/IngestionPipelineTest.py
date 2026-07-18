"""
Tests + testing NOTES for L1_DataProcessing/IngestionPipeline.py.

IngestionPipeline (OrderBookDashboard) is the ONLY layer that talks to the outside
world: it opens a WebSocket per venue, receives raw frames, runs them through a
payload extractor + normalizer, and stores {order_books, order_book_ts}. Everything
downstream (MultiVenueFeed.snapshot -> DataProcessing graph) is fed from those two
dicts.

That splits the module cleanly into two testable halves:

  1. The PURE transform half -- _normalize_symbol, _normalize_quote,
     _standardize_order_book, _default_payload_extractor, and the read side
     get_top_of_book / get_best_prices / get_quote_ts. These take a dict in and give
     a dict/tuple out; no socket, no thread, no time dependence (except the staleness
     clock, which we control). ALL of the active tests below cover this half.

  2. The LIVE dataflow half -- run() -> _start_async_loop() -> _listen(), which is an
     async reconnect loop over `websockets.connect`. This is where "the dataflow from
     API calls" actually lives, and it must NEVER hit the real network in a test
     (nondeterministic, rate-limited, offline-hostile). The big comment block near the
     bottom of this file is the requested write-up on HOW to test that half.

Run:  pytest TestCases/IngestionPipelineTest.py -v
"""

import os
import sys
import time

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from L1_DataProcessing.IngestionPipeline import OrderBookDashboard


def _dash(pairs=("ethbtc",), **kw):
    # Constructing the object does NOT start a thread -- run() does. So we can build
    # one freely and poke its pure methods / inject books by hand. "mock:" url just
    # keeps it from being mistaken for a live stream if someone later calls run().
    kw.setdefault("show", False)
    return OrderBookDashboard(list(pairs), "mock:test", **kw)


class TestNormalizeSymbol:
    def test_strips_punctuation_and_lowercases(self):
        d = _dash()
        assert d._normalize_symbol("ETH/BTC") == "ethbtc"
        assert d._normalize_symbol("BTC-USDT") == "btcusdt"
        assert d._normalize_symbol("  Sol_USD           ") == "solusd"

    def test_none_and_non_string(self):
        d = _dash()
        assert d._normalize_symbol(None) == ""
        assert d._normalize_symbol(123) == "123"


class TestNormalizeQuote:
    def test_list_form(self):
        assert _dash()._normalize_quote(["17.2", "5.3"]) == (17.2, 5.3)

    def test_dict_price_size(self):
        assert _dash()._normalize_quote({"price": "17.2", "size": "5.3"}) == (17.2, 5.3)

    def test_dict_alt_keys_default_size_zero(self):
        # only a bid key, no size -> size defaults to 0.
        assert _dash()._normalize_quote({"bid": "1.5"}) == (1.5, 0.0)

    def test_bad_inputs_return_none(self):
        d = _dash()
        assert d._normalize_quote("garbage") is None
        assert d._normalize_quote(["only-one"]) is None      # len < 2
        assert d._normalize_quote(["a", "b"]) is None         # non-numeric


class TestStandardizeOrderBook:
    def test_sorts_bids_desc_and_asks_asc(self):
        payload = {
            "bids": [["104999", "2"], ["105000", "1.2"]],
            "asks": [["105002", "1"], ["105001", "0.5"]],
        }
        out = _dash()._standardize_order_book(payload)
        assert out["bids"][0] == (105000.0, 1.2)   # best (highest) bid first
        assert out["asks"][0] == (105001.0, 0.5)   # best (lowest) ask first
        assert out["bids"][0] != [105000.0, 1.2]   
        assert out["asks"][0] != [105001.0, 0.5]

    def test_alt_keys_and_single_dict_side(self):
        payload = {"bid": {"price": "10", "size": "1"}, "ask": {"price": "11", "size": "1"}}
        out = _dash()._standardize_order_book(payload)
        assert out["bids"][0] == (10.0, 1.0)
        assert out["asks"][0] == (11.0, 1.0)

    def test_empty_or_non_dict_returns_none(self):
        d = _dash()
        assert d._standardize_order_book({"bids": [], "asks": []}) is None
        assert d._standardize_order_book("not a dict") is None


class TestDefaultPayloadExtractor:
    def test_binance_combined_stream_key_is_stripped(self):
        # "<symbol>@<channel>" -> bare symbol; otherwise normalization folds the
        # suffix in ("ethbtcdepth5100ms") and the exact-match lookup never finds it.
        data = {"stream": "ethbtc@depth5@100ms",
                "data": {"bids": [["1", "2"]], "asks": [["3", "4"]]}}
        key, payload = _dash()._default_payload_extractor(data)
        assert key == "ethbtc"
        assert payload == data["data"]

    def test_symbol_keyed_payload_without_data_wrapper(self):
        data = {"symbol": "BTCUSDT", "bids": [["1", "2"]], "asks": [["3", "4"]]}
        key, payload = _dash()._default_payload_extractor(data)
        assert key == "btcusdt"
        assert payload is data

    def test_non_dict_payload_returns_none(self):
        assert _dash()._default_payload_extractor({"data": "not-a-dict"}) is None


class TestReadSide:
    def _fresh(self, max_quote_age=None):
        d = _dash(["ethbtc"], max_quote_age=max_quote_age)
        d.order_books["ethbtc"] = {"bids": [["0.06", "1"]], "asks": [["0.061", "2"]]}
        d.order_book_ts["ethbtc"] = time.time()
        return d

    def test_get_top_of_book_returns_best_levels(self):
        bid, bid_size, ask, ask_size = self._fresh().get_top_of_book("ethbtc")
        assert (bid, bid_size, ask, ask_size) == (0.06, 1.0, 0.061, 2.0)

    def test_missing_pair_is_na(self):
        assert self._fresh().get_top_of_book("xrpbtc") == ("N/A", "N/A", "N/A", "N/A")

    def test_exact_match_only_no_substring_bleed(self):
        # a btcusdt book must NOT answer a btcusd request (substring bug would).
        d = _dash(["btcusdt"])
        d.order_books["btcusdt"] = {"bids": [["100", "1"]], "asks": [["101", "1"]]}
        d.order_book_ts["btcusdt"] = time.time()
        assert d.get_top_of_book("btcusd") == ("N/A", "N/A", "N/A", "N/A")

    def test_stale_quote_dropped_under_max_quote_age(self):
        d = self._fresh(max_quote_age=0.5)
        d.order_book_ts["ethbtc"] = time.time() - 10     # 10s old, window 0.5s
        assert d.get_top_of_book("ethbtc") == ("N/A", "N/A", "N/A", "N/A")
        assert d.get_quote_ts("ethbtc") is None

    def test_fresh_quote_reports_ts(self):
        d = self._fresh(max_quote_age=5.0)
        assert d.get_quote_ts("ethbtc") is not None
        assert d.get_best_prices("ethbtc") == (0.06, 0.061)

class FakeWebSocket:

    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []

    async def send(self, msg):
        self.sent.append(msg)

    async def recv(self):
        if self.messages:
            return self.messages.pop(0)

        raise Exception("disconnect")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False
    
from unittest.mock import AsyncMock, MagicMock, patch
import json
class TestListen:
    
    @pytest.mark.asyncio
    async def test_initial_subscription_sent():
        fake_ws = AsyncMock()
        fake_ws.recv.side_effect = Exception("stop")

        fake_connect = AsyncMock()
        fake_connect.__aenter__.return_value = fake_ws

        feed = MagicMock()

        feed.stream_url = "ws://localhost"
        feed.is_running = True
        feed.debug = False
        feed.initial_message = {"type": "subscribe"}
        feed.payload_extractor = lambda x: None
        feed._default_payload_extractor = None
        feed.order_books = {}
        feed.order_book_ts = {}
        feed.broker_name = "TEST"

        async def stop_sleep(*args):
            feed.is_running = False

        with patch("your_module.websockets.connect", return_value=fake_connect), \
            patch("your_module.asyncio.sleep", side_effect=stop_sleep):

            await YourClass._listen(feed)

        fake_ws.send.assert_awaited_once_with(
            json.dumps({"type": "subscribe"})
        )
        


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
