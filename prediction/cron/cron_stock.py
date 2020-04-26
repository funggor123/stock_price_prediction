import yfinance as yf
import pandas as pd
import os.path

yf.pdr_override()  # <== that's all it takes :-)
from pandas_datareader import data as pdr
from prediction.cron.list_xls_csv import get_stock_list

# Start and Exlucde End

# yfinance usages
# ref: https://aroussi.com/post/python-yahoo-finance

# fetch a stock from yahoo finance using its stock code
# input: stock_code string
#        data_start date time object
#        data_end   date time object
#        data_interval string (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
#        data_period string (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
# return: pandas preprocess frame object
def fetch_stock(data_start="2003-12-01", data_end="2020-04-01", stock_code="0005.hk", cache_stock_data_path=None):
    if cache_stock_data_path is not None:
        if os.path.isfile(cache_stock_data_path):
            return pd.read_csv(cache_stock_data_path, index_col=0)
    df = pdr.get_data_yahoo(stock_code, start=data_start, end=data_end)
    if df['Close'].count() != 0:
        df.to_csv('../resources/raw/stock_data_' + stock_code + '_' + data_start + '_' + data_end + '.csv')
        return df
    else:
        return None


def fetch_stocks(data_start="2003-12-01", data_end="2020-04-01", stock_list_path="../resources/industries/", use_cache=False):
    df = get_stock_list(stock_list_path, None)
    stocks_data, stocks_code, fail_cron_list = [], [], []
    for index, row in df.iterrows():
        print("fetching  " + row['Stock_Code'])
        print('current_index: ' + str(index))

        if use_cache:
            cache_stock_data_path = '../resources' + '/raw/stock_data_' + row['Stock_Code'] + '_' + data_start + '_' + data_end + '.csv'
        else:
            cache_stock_data_path = None

        stock_data = fetch_stock(data_start, data_end, row['Stock_Code'], cache_stock_data_path=cache_stock_data_path)
        if stock_data is not None:
            stocks_data.append(stock_data)
            stocks_code.append(row['Stock_Code'])
        else:
            fail_cron_list.append(row['Stock_Code'])
    return stocks_data, stocks_code, fail_cron_list


# TODO method of fetching related stocks from yahoo finance
def fetch_related_stocks():
    return None
