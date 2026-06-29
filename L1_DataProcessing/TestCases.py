import pytest

try:
    from DataProcessing import ExchangeRateGraph
    from IngestionPipeline import OrderBookDashboard
    from MultiVenueFeed import MultiBrokerOrderBook
except:
    from .DataProcessing import ExchangeRateGraph
    from .IngestionPipeline import OrderBookDashboard
    from .MultiVenueFeed import MultiBrokerOrderBook
    
def test_ExchangeRateGraph():
    