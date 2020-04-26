from prediction.preprocess.technical_analysis import get_technical_indicators
from prediction.cron.cron_stock import fetch_stock
import pandas as pd


def test_preprocess():
    stock_data = fetch_stock()
    stock_data = get_technical_indicators(stock_data)
    pd.set_option('display.max_columns', None)
    print(stock_data.columns)
    print(stock_data.tail())


test_preprocess()
