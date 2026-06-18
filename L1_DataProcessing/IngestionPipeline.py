import asyncio
import threading
import json
import time
from itertools import combinations

try:
    import websockets
except Exception:
    websockets = None

class OrderBookDashboard:
    def __init__(self, pairs, stream_url, broker_name="Binance", refresh_interval=0.1, show=True):
        self.pairs = pairs
        self.refresh_interval = refresh_interval
        self.order_books = {}
        self.is_running = True
        self.stream_url = stream_url
        self.broker_name = broker_name
        self.show = show

    async def _listen(self):
        """acts like the receiver, as data arrives, it parses to JSON and updats the order_books right the moment"""
        if websockets is None:
            # If websockets not installed, raise a clear error when attempting to listen
            raise RuntimeError("websockets package is required for _listen()")

        async with websockets.connect(self.stream_url) as ws:
            while self.is_running:
                try:
                    raw_data = await ws.recv()
                    data = json.loads(raw_data)
                    payload = data.get("data")
                    if payload:
                        self.order_books[data.get("stream")] = payload
                except Exception:
                    await asyncio.sleep(1)

    def _start_async_loop(self):
        """this is here because we want to run _listen while executes the loops"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._listen())

    def get_best_prices(self, pair_name):
        key = next((k for k in self.order_books if pair_name.lower() in k), None)
        if key:
            bids = self.order_books[key].get('bids', [])
            asks = self.order_books[key].get('asks', [])
            return (bids[0][0] if bids else "N/A"), (asks[0][0] if asks else "N/A")
        return "N/A", "N/A"

    def run(self):
        """Starts the background thread and the display loop."""
        threading.Thread(target=self._start_async_loop, daemon=True).start() #simultaneously running the wholething while
        #connects the loop of data catching
        
        if self.show:
            try:
                print("Dashboard initialized...")
                while self.is_running:
                    if not self.order_books:
                        time.sleep(0.1)
                        continue

                    print("\033[H\033[J", end="") #print the times
                    print(f"--- Live Market ({time.strftime('%H:%M:%S')}) ---")
                    
                    for p in self.pairs:
                        bid, ask = self.get_best_prices(p)
                        print(f"{p.upper():<10} | Bid: {bid:<12} | Ask: {ask:<12}")
                    
                    time.sleep(self.refresh_interval)
            except KeyboardInterrupt:
                self.is_running = False
                print("\nShutting down.")
            
        

if __name__ == "__main__":
    # Quote priority: if an asset appears earlier in this list, it's used
    # as the quote currency against assets that appear later (matches
    # Binance's actual symbol naming, e.g. ETHBTC not BTCETH).
    quote_priority = ["btc", "eth", "bnb", "sol"]
    assets = ["btc", "eth", "bnb", "sol"]

    def make_pair(a, b):
        base, quote = (b, a) if quote_priority.index(a) < quote_priority.index(b) else (a, b)
        return f"{base}{quote}"

    # K4: every asset directly connected to every other asset (6 edges, no hub)
    my_pairs = [make_pair(a, b) for a, b in combinations(assets, 2)]
    # -> ['ethbtc', 'bnbbtc', 'solbtc', 'bnbeth', 'soleth', 'solbnb']

    refresh_rate = 0.05  # 50ms updates
    url = (
        "wss://stream.binance.com:9443/stream?streams="
        + "/".join(f"{p}@depth5@100ms" for p in my_pairs)
    )

    dashboard = OrderBookDashboard(my_pairs, url, refresh_interval=refresh_rate)
    dashboard.run()