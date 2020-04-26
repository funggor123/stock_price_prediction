import yfinance as yf
import os.path
import pandas as pd

def rename_yahoo_finance_df_columns(df_obj, prefix):
    df_obj.columns = [prefix + "_" + col for col in df_obj.columns]


# Major Stock Indices
# ref: https://markets.businessinsider.com/indices
# US: Dow Jones, S&P 500, NASDAQ Composite
# Asia: HANG SENG INDEX, Nikkei 225, S&P BSE SENSEX, SSE Composite Index
# Euro: FTSE 100, CAC 40, DAX
COMPOSITE_INDICES_LIST = [
    "^IXIC",
    "^GSPC",
    "^DJI",
    "^HSI",
    "^N225",
    "^BSESN",
    "^FTSE",
    "^FCHI",
    "^GDAXI"
    # "^SHSS" (missing data),
]


# fetch major composite indices from yahoo finance
# input: data_start date time object
#        data_end   date time object
#        data_interval string (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
#        data_period string (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
# return: pandas preprocess frame object
def fetch_composite_indices(data_start="2003-12-01", data_end="2020-04-01", data_period="max",
                            data_interval="1d", use_cache=False):
    indices_df_list = []
    for index in COMPOSITE_INDICES_LIST:
        index_obj = yf.Ticker(index)
        df = None
        cache_composite_path = '../resources/raw/composite_data_' + index + '_' + data_start + '_' + data_end + '.csv'
        if use_cache is True:
            if os.path.isfile(cache_composite_path):
                df = pd.read_csv(cache_composite_path, index_col=0)
                indices_df_list.append(df)
                continue
        if data_start is not None and data_end is not None:
            df = index_obj.history(start=data_start, end=data_end, interval=data_interval)
        else:
            df = index_obj.history(period=data_period, interval=data_interval)
        rename_yahoo_finance_df_columns(df, index)
        if df[index + "_" + 'Close'].count() != 0:
            df.to_csv(cache_composite_path)
        indices_df_list.append(df)
    return indices_df_list


# Major Currencies
# Top ten other currencies + RMB to HKD
CURRENCY_LIST = [
    "USDHKD=X",
    "EURHKD=X",
    "GBPHKD=X",
    "AUDHKD=X",
    "CADHKD=X",
    "SGDHKD=X",
    # "CHFHKD=X" (missing data from 2004-2005),
    "MYRHKD=X",
    # "JPYHKD=X" (date start from 2006),
    "CNYHKD=X",
]


# fetch major currencies from yahoo finance
# input: data_start date time object
#        data_end   date time object
#        data_interval string (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
#        data_period string (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
# return: pandas preprocess frame object
def fetch_currencies(data_start="2003-12-01", data_end="2020-04-03", data_period="max",
                     data_interval="1d", use_cache=False):
    currencies_df_list = []
    for currency in CURRENCY_LIST:
        currency_obj = yf.Ticker(currency)
        df = None
        cache_composite_path = '../resources/raw/currency_' + currency + '_' + data_start + '_' + data_end + '.csv'
        if use_cache is True:
            if os.path.isfile(cache_composite_path):
                df = pd.read_csv(cache_composite_path, index_col=0)
                currencies_df_list.append(df)
                continue
        if data_start is not None and data_end is not None:
            df = currency_obj.history(start=data_start, end=data_end, interval=data_interval)
        else:
            df = currency_obj.history(period=data_period, interval=data_interval)
        rename_yahoo_finance_df_columns(df, currency)
        if df[currency + "_" + 'Close'].count() != 0:
            df.to_csv(cache_composite_path)
        currencies_df_list.append(df)
    return currencies_df_list


# TODO fetch global economy indices
def fetch_global_economy_indices(data_start=None, data_end=None, data_period="max",
                                 data_interval="1d"):
    return None


