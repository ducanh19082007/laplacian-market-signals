import pytest

try:
    from DataProcessing import ExchangeRateGraph
    from IngestionPipeline import OrderBookDashboard
    from MultiVenueFeed import MultiBrokerOrderBook
except:
    from ..L1_DataProcessing.DataProcessing import ExchangeRateGraph
    from ..L1_DataProcessing.IngestionPipeline import OrderBookDashboard
    from ..L1_DataProcessing.MultiVenueFeed import MultiBrokerOrderBook
    
def test_ExchangeRateGraph():
    