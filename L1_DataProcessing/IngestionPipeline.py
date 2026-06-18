import asyncio
import threading
import json
import time
import websockets

class OrderBookDashboard:
    def __init__(self, pairs, refresh_interval=0.1):
        self.pairs = pairs
        self.refresh_interval = refresh_interval
        self.order_books = {}
        self.is_running = True
        self.stream_url = (
            "wss://stream.binance.com:9443/stream?streams="
            + "/".join(f"{p}@depth5@100ms" for p in self.pairs)
        )

    async def _listen(self):
        async with websockets.connect(self.stream_url) as ws:
            while self.is_running:
                try:
                    raw_data = await ws.recv()
                    data = json.loads(raw_data)
                    payload = data.get("data")
                    if payload:
                        self.order_books[data.get("stream")] = payload
                except Exception as e:
                    await asyncio.sleep(1)

    def _start_async_loop(self):
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
        threading.Thread(target=self._start_async_loop, daemon=True).start()
        
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
            
            
# Configuration
my_pairs = ["btcusdt", "ethusdt", "bnbusdt", "solusdt", "xrpusdt", "dogeusdt", "adausdt"]
refresh_rate = 0.05 # 50ms updates

# Initialize and Run
dashboard = OrderBookDashboard(my_pairs, refresh_interval=refresh_rate)
dashboard.run()