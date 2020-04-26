from prediction.cron.cron_stock import fetch_stock
from prediction.cron.cron_correlated_assets import fetch_correlated_assets
import pandas as pd


def test_cron():
    stock_data = fetch_stock()
    correlated_assets = fetch_correlated_assets()
    pd.set_option('display.max_columns', None)
    print(correlated_assets.columns)
    print(correlated_assets.tail())


test_cron()
