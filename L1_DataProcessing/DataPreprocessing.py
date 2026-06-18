import time
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Any


try:
    from .IngestionPipeline import OrderBookDashboard
except ImportError:
    from IngestionPipeline import OrderBookDashboard







# NOTE: Binance is configured to use a real websocket depth stream.
# OANDA and IBKR are currently configured as mock feeds because this code
# does not yet include real broker-specific endpoints, authentication, or
# payload parsing for their live market data.
#
# To use OANDA/IBKR live, you must replace the mock URLs with real broker
# stream URLs and add dedicated payload extractors for their feed formats.
#Author: Anh Duc Le









@dataclass
class BrokerConfig:
    name: str
    stream_url: str
    pairs: List[str]
    payload_extractor: Optional[Any] = None
    debug: bool = False


class MultiBrokerOrderBook:
    def __init__(self, brokers: List[BrokerConfig], refresh_interval: float = 0.5):
        self.dashboards = []
        self.refresh_interval = refresh_interval

        for broker in brokers:
            dashboard = OrderBookDashboard(
                broker.pairs,
                broker.stream_url,
                broker_name=broker.name,
                refresh_interval=refresh_interval,
                show=False,
                payload_extractor=broker.payload_extractor,
                debug=broker.debug,
            )
            dashboard.run()
            self.dashboards.append((broker, dashboard))

    def get_all_pairs(self) -> List[str]:
        pairs = set()
        for broker, _ in self.dashboards:
            pairs.update(pair.lower() for pair in broker.pairs)
        return sorted(pairs)

    def snapshot(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        snapshot = {}
        for pair in self.get_all_pairs():
            snapshot[pair] = {}
            for broker, dashboard in self.dashboards:
                bid, ask = dashboard.get_best_prices(pair)
                snapshot[pair][broker.name] = {"bid": bid, "ask": ask}
        return snapshot

    def aggregate_best_prices(self, pair_name: str) -> Dict[str, str]:
        bids = []
        asks = []
        for _, dashboard in self.dashboards:
            bid, ask = dashboard.get_best_prices(pair_name)
            try:
                bids.append(float(bid))
            except (ValueError, TypeError):
                pass
            try:
                asks.append(float(ask))
            except (ValueError, TypeError):
                pass

        return {
            "best_bid": max(bids) if bids else "N/A",
            "best_ask": min(asks) if asks else "N/A",
        }

    def print_table(self) -> None: #NOTE: this is suggested with LLMs so its design is a bit... 
        brokers = [broker.name for broker, _ in self.dashboards]
        header = ["PAIR"] + [f"{name} BID" for name in brokers] + [f"{name} ASK" for name in brokers] + ["BEST BID", "BEST ASK"]
        column_width = 14

        print(" | ".join(h.upper().ljust(column_width) for h in header))
        print("-" * (len(header) * (column_width + 3)))

        snapshot = self.snapshot()
        for pair in self.get_all_pairs():
            row = [pair.upper().ljust(column_width)]
            for broker_name in brokers:
                row.append(str(snapshot[pair][broker_name]["bid"]).ljust(column_width))
            for broker_name in brokers:
                row.append(str(snapshot[pair][broker_name]["ask"]).ljust(column_width))
            aggregated = self.aggregate_best_prices(pair)
            row.append(str(aggregated["best_bid"]).ljust(column_width))
            row.append(str(aggregated["best_ask"]).ljust(column_width))
            print(" | ".join(row))

    def run_live(self) -> None:
        try:
            while True:
                print("\033[H\033[J", end="")
                print(f"--- Multi-Broker Order Book ({time.strftime('%H:%M:%S')}) ---")
                self.print_table()
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            for _, dashboard in self.dashboards:
                dashboard.is_running = False
            print("\nMulti-broker monitoring stopped.")


def make_binance_depth_url(pairs: List[str], depth: int = 5, interval: str = "100ms") -> str:
    return (
        "wss://stream.binance.com:9443/stream?streams="
        + "/".join(f"{p}@depth{depth}@{interval}" for p in pairs)
    )


if __name__ == "__main__":
    quote_priority = ["btc", "eth", "bnb", "sol"]
    assets = ["btc", "eth", "bnb", "sol"]

    def make_pair(a: str, b: str) -> str:
        base, quote = (b, a) if quote_priority.index(a) < quote_priority.index(b) else (a, b)
        return f"{base}{quote}"

    my_pairs = [make_pair(a, b) for a, b in combinations(assets, 2)]

    binance = BrokerConfig(
        name="Binance",
        stream_url=make_binance_depth_url(my_pairs),
        pairs=my_pairs,
        payload_extractor=None,
    )

    oanda = BrokerConfig(
        name="OANDA",
        stream_url="mock:oanda",
        pairs=[pair.lower() for pair in my_pairs],
        payload_extractor=None,
    )

    ibkr = BrokerConfig(
        name="IBKR",
        stream_url="mock:ibkr",
        pairs=[pair.lower() for pair in my_pairs],
        payload_extractor=None,
    )

    broker_configs = [binance, oanda, ibkr]

    multi_broker = MultiBrokerOrderBook(broker_configs, refresh_interval=0.5)
    multi_broker.run_live()
