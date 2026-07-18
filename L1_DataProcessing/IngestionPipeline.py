import asyncio
import threading
import json
import random
import re
import time
from itertools import combinations
from typing import Any, Callable, Dict, Optional, Tuple

try:
    import websockets
except Exception:
    websockets = None
    
#Author: Anh Duc Le

PayloadExtractor = Callable[[Dict[str, Any]], Optional[Tuple[str, Dict[str, Any]]]]


class OrderBookDashboard:
    def __init__(self,
        pairs,
        stream_url,
        broker_name="Binance",
        refresh_interval=0.1,
        show=True,
        payload_extractor: Optional[PayloadExtractor] = None,
        initial_message: Optional[Any] = None,
        debug=True,
        max_quote_age: Optional[float] = None,
    ):
        self.pairs = [self._normalize_symbol(p) for p in pairs]
        self.refresh_interval = refresh_interval
        self.order_books: Dict[str, Dict[str, Any]] = {}
        # Wall-clock time each book key was last updated, for stale data checks.
        # A venue only ticks when a WS message arrives, so without this a lagging
        # venue keeps serving a stale quote that shows up as phantom arbitrage. hence
        # this could creates something similar to being hallucinate and the program
        # got tricked by their own fetched data,
        self.order_book_ts: Dict[str, float] = {}
        # Quotes older than this (seconds) are treated as N/A. None = never expire.
        # and also the order_book and order_book_ts is our output
        self.max_quote_age = max_quote_age
        self.is_running = True
        self.stream_url = stream_url
        self.broker_name = broker_name
        self.show = show
        self.debug = debug
        self.payload_extractor = payload_extractor
        self.initial_message = initial_message
        #this two only works for mock data/url, one can skip this and pays no worries if
        #one doesnt use live-mock data
        self.mock = isinstance(stream_url, str) and stream_url.startswith("mock:")
        self.mock_seed = stream_url.split(":", 1)[1] if self.mock else broker_name

    def _normalize_symbol(self, symbol: Any) -> str:
        #for any strings, this will cleans all the redundancies such as extra space, lowercase all letters,..
        if symbol is None:
            return ""
        return re.sub(r"[^a-z0-9]", "", str(symbol).lower())

    def _normalize_quote(self, quote: Any) -> Optional[Tuple[float, float]]:
        # This function aims to normalize a quote into a standard (price, size) format.
        #EX: ["17.2", "5.3"] or {"price": "17.2", "size": "5.3"}
        #we will split it out distinctively where the first value will be the price
        #of the ask/bid and the second will be the size of it
        # For list/tuple quotes, assume:
        # quote[0] = price
        # quote[1] = size
        
        #OR, we will input a dictionary right away, where one will go through all the possible
        #key names that could be named differently by different broker's preference on their data.
        # {"price": "17.2", "size": "5.3"}
        if isinstance(quote, (list, tuple)) and len(quote) >= 2:
            try:
                return float(quote[0]), float(quote[1]) #str -> float 
            except Exception:
                return None
        if isinstance(quote, dict):
            price = quote.get("price") or quote.get("p") or quote.get("bid") or quote.get("ask")
            size = quote.get("size") or quote.get("volume") or quote.get("liquidity") or 0
            try:
                return float(price), float(size)
            except Exception:
                return None
        return None

    def _standardize_order_book(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Inputs:
        payload = {
            "lastUpdateId": 12345,
            "bids": [["105000", "1.2"], ...],
            "asks": [ ["105001", "0.5"], ...]
        }
        
        Outputs: 
        {
            "bids": [(105000.0, 1.2),...],
            "asks": [(105001.0, 0.5), ...]
        }
        """
        if not isinstance(payload, dict):
            return None

        bids = payload.get("bids") or payload("bid") or payload.get("buy") or []
        asks = payload.get("asks") or payload.get("ask") or payload.get("sell") or []

        if isinstance(bids, dict):
            bids = [bids]
        if isinstance(asks, dict):
            asks = [asks]

        normalized_bids = [q for q in (self._normalize_quote(q) for q in bids) if q is not None] #this also help change str to float
        normalized_asks = [q for q in (self._normalize_quote(q) for q in asks) if q is not None]

        if not normalized_bids and not normalized_asks:
            return None

        normalized_bids.sort(key=lambda x: x[0], reverse=True)
        normalized_asks.sort(key=lambda x: x[0])

        return {"bids": normalized_bids, "asks": normalized_asks}

    def _default_payload_extractor(self, data: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
        #this function is change Raw WebSocket Message into (symbol, payload)
        #for instance, for Binance
        """
        data = {
            "stream": "ethbtc@depth5@100ms",
            "data": {
                "bids": [["0.0345", "100"]],
                "asks": [["0.0346", "50"]]
            }
        }
        """
        payload = data.get("data") or data
        if not isinstance(payload, dict):
            return None
        key = data.get("stream") or data.get("symbol") or data.get("instrument") or ""
        # Binance combined streams name the channel "<symbol>@<channel>"
        # (e.g. "ethbtc@depth5@100ms"). Strip everything from the first "@"
        # so the stored key is the bare symbol -- otherwise normalization folds
        # the suffix in ("ethbtcdepth5100ms") and the exact-match lookup in
        # get_top_of_book never finds it. Venues that key by plain
        # symbol/instrument have no "@" and are unaffected. Just for precaution, we have this
        #little function
        if isinstance(key, str):
            key = key.split("@", 1)[0] #ethbtc
        return self._normalize_symbol(key), payload

    async def _listen(self):
        
        """
        while running:
        connect websocket

        send subscription

        while connected:
            receive message
            extract payload
            store order book
            store timestamp

        if disconnected:
            reconnect
        """
        if websockets is None:
            raise RuntimeError("websockets package is required for _listen()")

        while self.is_running:
            try:
                async with websockets.connect(self.stream_url) as ws:
                    
                    if self.debug:
                        print(f"[{self.broker_name}] connected to {self.stream_url}")
                    if self.initial_message is not None:
                        # A list means "send each message in turn" -- some venues/brokers
                        # (e.g. Bitstamp) require one subscribe frame per channel,
                        # while others (Kraken/OKX/Coinbase/Gemini) batch every
                        # symbol into a single frame. Both are handled here.
                        messages = (self.initial_message if isinstance(self.initial_message, list) else [self.initial_message])
                        for msg in messages:
                            await ws.send(json.dumps(msg))
                            if self.debug:
                                print(f"[{self.broker_name}] sent initial subscription: {msg}")
                                
                    while self.is_running:
                        try:
                            raw_data = await ws.recv() #await until receive the raw message
                            data = json.loads(raw_data) #after that, loads the received message into JSON
                            extractor = self.payload_extractor or self._default_payload_extractor #either uses the onei build or the one
                            #available in MultiVenueFeed.py URLmethods methods
                            result = extractor(data)
                            if result is not None:
                                key, payload = result
                                normalized_key = self._normalize_symbol(key)
                                self.order_books[normalized_key] = payload #store order book
                                self.order_book_ts[normalized_key] = time.time() #store timestamp
                                #such that the two dictionary that have normalized_key keys corresponds
                            elif self.debug:
                                print(f"[{self.broker_name}] ignored message: {data}")
                        except Exception as exc:
                            if self.debug:
                                print(f"[{self.broker_name}] receive error: {repr(exc)}")
                            await asyncio.sleep(1)
            except Exception as exc:
                if self.debug:
                    print(f"[{self.broker_name}] connect error: {repr(exc)}")
                await asyncio.sleep(5)

    #dont have to worry about this if one doesn't want to use mock data
    def _mock_loop(self):
        rng = random.Random(self.mock_seed)
        base_prices = {pair: rng.uniform(0.5, 1.5) for pair in self.pairs}
        while self.is_running:
            for pair in self.pairs:
                mid = base_prices[pair]
                spread = rng.uniform(0.0001, 0.001)
                bid = round(mid - spread / 2, 6)
                ask = round(mid + spread / 2, 6)
                self.order_books[pair] = {
                    "bids": [[bid, round(rng.uniform(1, 10), 4)]],
                    "asks": [[ask, round(rng.uniform(1, 10), 4)]],
                }
            time.sleep(self.refresh_interval)

    def _start_async_loop(self):
        #either do a mock data or actual data, using asyncio to create a new inf event loop and grab raw data
        if self.mock:
            self._mock_loop()
            return
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._listen())

    def get_top_of_book(self, pair_name):
        """
        Best bid/ask AND their sizes, honoring the same exact-match + max_quote_age
        rules as get_best_prices. Returns (bid, bid_size, ask, ask_size); any field
        that has no fresh quote is "N/A" tp control stalement.

        Size is what lets the arbitrage graph reject a mispriced top-of-book that's
        only good for a trivial amount -- the dominant source of phantom arbitrage on
        thin markets -- so it's surfaced here rather than discarded.
        """
        # Exact symbol match only. The old substring match (`target in k or
        # k in target`) would serve a USDT/USDC book for a USD request and vice
        # versa -- "btcusd" is a substring of "btcusdt" -- feeding the wrong
        # price into a convert edge and fabricating arbitrage. Stored keys and
        # `target` are both normalized, so equality is the correct test.
        target = self._normalize_symbol(pair_name)
        if target not in self.order_books:
            return "N/A", "N/A", "N/A", "N/A"
        # Drop quotes that haven't refreshed within max_quote_age: a stale book
        # crossed against a fresher venue is the main source of phantom arbitrage.
        if self.max_quote_age is not None:
            age = time.time() - self.order_book_ts.get(target, 0.0)
            if age > self.max_quote_age:
                return "N/A", "N/A", "N/A", "N/A"
        standardized = self._standardize_order_book(self.order_books[target]) #finishes standardize the data
        #after this, one just need to fetch the top of the order book for the best bid and ask price alongside 
        #with their size
        if standardized is None:
            if self.debug:
                print(f"[{self.broker_name}] unsupported payload for key={target}: {self.order_books[target]}")
            return "N/A", "N/A", "N/A", "N/A"
        bids = standardized.get("bids", [])
        asks = standardized.get("asks", [])
        bid, bid_size = (bids[0][0], bids[0][1]) if bids else ("N/A", "N/A")
        ask, ask_size = (asks[0][0], asks[0][1]) if asks else ("N/A", "N/A")
        return bid, bid_size, ask, ask_size

    def get_best_prices(self, pair_name):
        bid, _bid_size, ask, _ask_size = self.get_top_of_book(pair_name)
        return bid, ask

    def get_quote_ts(self, pair_name) -> Optional[float]:
        """
        Wall-clock time the book for pair_name last updated, or None if there's
        no (sufficiently fresh) quote.

        Mirrors get_best_prices' exact-match + max_quote_age rule so a quote that
        reads N/A there reports no timestamp here. Used by the feed to stamp each
        snapshot quote, which the graph's contemporaneity guard relies on.
        """
        target = self._normalize_symbol(pair_name)
        if target not in self.order_books:
            return None
        ts = self.order_book_ts.get(target, 0.0)
        if self.max_quote_age is not None and (time.time() - ts) > self.max_quote_age: #check either if we want a
            #limit to the dead data or no and see if they are dead or not
            return None
        return ts

    def run(self):
        threading.Thread(target=self._start_async_loop, daemon=True).start() #threading, running different 
        #broker's own IngestionPipeline simultaneously in MultiVenueFeed
        if self.show:
            try:
                print("Dashboard initialized...")
                while self.is_running:
                    if not self.order_books:
                        time.sleep(0.1)
                        continue

                    print("\033[H\033[J", end="")
                    print(f"--- Live Market ({time.strftime('%H:%M:%S')}) ---")
                    for p in self.pairs:
                        bid, ask = self.get_best_prices(p)
                        print(f"{p.upper():<10} | Bid: {bid:<12} | Ask: {ask:<12}")
                    time.sleep(self.refresh_interval)
            except KeyboardInterrupt:
                self.is_running = False
                print("\nShutting down.")
                
    @staticmethod
    def make_pair(a, b):
        base, quote = (b, a) if quote_priority.index(a) < quote_priority.index(b) else (a, b)
        return f"{base}{quote}"


if __name__ == "__main__":
    #test case
    quote_priority = ["btc", "eth", "bnb", "sol"]
    assets = ["btc", "eth", "bnb", "sol"]

    my_pairs = [OrderBookDashboard.make_pair(a, b) for a, b in combinations(assets, 2)]

    refresh_rate = 0.05
    url = (
        "wss://stream.binance.com:9443/stream?streams="
        + "/".join(f"{p}@depth5@100ms" for p in my_pairs)
    )

    dashboard = OrderBookDashboard(my_pairs, url, refresh_interval=refresh_rate)
    dashboard.run()
