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
    def __init__(
        self,
        pairs,
        stream_url,
        broker_name="Binance",
        refresh_interval=0.1,
        show=True,
        payload_extractor: Optional[PayloadExtractor] = None,
        initial_message: Optional[Any] = None,
        debug=False,
    ):
        self.pairs = [self._normalize_symbol(p) for p in pairs]
        self.refresh_interval = refresh_interval
        self.order_books: Dict[str, Dict[str, Any]] = {}
        self.is_running = True
        self.stream_url = stream_url
        self.broker_name = broker_name
        self.show = show
        self.debug = debug
        self.payload_extractor = payload_extractor
        self.initial_message = initial_message
        self.mock = isinstance(stream_url, str) and stream_url.startswith("mock:")
        self.mock_seed = stream_url.split(":", 1)[1] if self.mock else broker_name

    def _normalize_symbol(self, symbol: Any) -> str:
        if symbol is None:
            return ""
        return re.sub(r"[^a-z0-9]", "", str(symbol).lower())

    def _normalize_quote(self, quote: Any) -> Optional[Tuple[float, float]]:
        if isinstance(quote, (list, tuple)) and len(quote) >= 2:
            try:
                return float(quote[0]), float(quote[1])
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
        if not isinstance(payload, dict):
            return None

        bids = payload.get("bids") or payload.get("bid") or payload.get("buy") or []
        asks = payload.get("asks") or payload.get("ask") or payload.get("sell") or []

        if isinstance(bids, dict):
            bids = [bids]
        if isinstance(asks, dict):
            asks = [asks]

        normalized_bids = [q for q in (self._normalize_quote(q) for q in bids) if q is not None]
        normalized_asks = [q for q in (self._normalize_quote(q) for q in asks) if q is not None]

        if not normalized_bids and not normalized_asks:
            return None

        normalized_bids.sort(key=lambda x: x[0], reverse=True)
        normalized_asks.sort(key=lambda x: x[0])

        return {"bids": normalized_bids, "asks": normalized_asks}

    def _default_payload_extractor(self, data: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
        payload = data.get("data") or data
        if not isinstance(payload, dict):
            return None
        key = data.get("stream") or data.get("symbol") or data.get("instrument") or ""
        return self._normalize_symbol(key), payload

    async def _listen(self):
        if websockets is None:
            raise RuntimeError("websockets package is required for _listen()")

        while self.is_running:
            try:
                async with websockets.connect(self.stream_url) as ws:
                    if self.debug:
                        print(f"[{self.broker_name}] connected to {self.stream_url}")
                    if self.initial_message is not None:
                        await ws.send(json.dumps(self.initial_message))
                        if self.debug:
                            print(f"[{self.broker_name}] sent initial subscription: {self.initial_message}")
                    while self.is_running:
                        try:
                            raw_data = await ws.recv()
                            data = json.loads(raw_data)
                            extractor = self.payload_extractor or self._default_payload_extractor
                            result = extractor(data)
                            if result is not None:
                                key, payload = result
                                normalized_key = self._normalize_symbol(key)
                                self.order_books[normalized_key] = payload
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
        if self.mock:
            self._mock_loop()
            return
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._listen())

    def get_best_prices(self, pair_name):
        target = self._normalize_symbol(pair_name)
        key = next((k for k in self.order_books if target in k or k in target), None)
        if key:
            standardized = self._standardize_order_book(self.order_books[key])
            if standardized is None:
                if self.debug:
                    print(f"[{self.broker_name}] unsupported payload for key={key}: {self.order_books[key]}")
                return "N/A", "N/A"
            bids = standardized.get("bids", [])
            asks = standardized.get("asks", [])
            return (bids[0][0] if bids else "N/A"), (asks[0][0] if asks else "N/A")
        return "N/A", "N/A"

    def run(self):
        threading.Thread(target=self._start_async_loop, daemon=True).start()
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


if __name__ == "__main__":
    quote_priority = ["btc", "eth", "bnb", "sol"]
    assets = ["btc", "eth", "bnb", "sol"]

    def make_pair(a, b):
        base, quote = (b, a) if quote_priority.index(a) < quote_priority.index(b) else (a, b)
        return f"{base}{quote}"

    my_pairs = [make_pair(a, b) for a, b in combinations(assets, 2)]

    refresh_rate = 0.05
    url = (
        "wss://stream.binance.com:9443/stream?streams="
        + "/".join(f"{p}@depth5@100ms" for p in my_pairs)
    )

    dashboard = OrderBookDashboard(my_pairs, url, refresh_interval=refresh_rate)
    dashboard.run()
