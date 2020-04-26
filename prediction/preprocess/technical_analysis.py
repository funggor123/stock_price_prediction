import pandas as pd

# Get stock's simple moving average
# input: stock_data     pandas data frame object
#        n              int (rolling period)
# return: pandas data frame object
def get_simple_moving_average(stock_data, n=7):
    return pd.Series(stock_data["Close"]).rolling(window=n).mean()


# pandas ewm
# ref: https://blog.csdn.net/weixin_41494909/article/details/99670246
# ref: https://blog.csdn.net/yanpenggong/article/details/84031655


# Get stock's exponential moving average
# input: stock_data     pandas data frame object
#        span           int (rolling period)
# return: pandas data frame object
def get_exp_moving_average(stock_data, span=12):
    return pd.Series(stock_data["Close"]).ewm(span=span).mean()


def get_exp_moving_average2(stock_data, com=0.5):
    return pd.Series(stock_data["Close"]).ewm(com=0.5).mean()


# momentum
# ref: https://www.investopedia.com/articles/technical/081501.asp

# Get stock's momentum
# input: stock_data     pandas data frame object
#        n              int (rolling period)
# return: pandas data frame object
def get_momentum(stock_data, n=7):
    return pd.Series(stock_data["Close"]).rolling(window=n).apply(diff)


def diff(x):
    return x[-1] - x[0]


# wiki macd
# ref: https://zh.wikipedia.org/wiki/%E6%8C%87%E6%95%B0%E5%B9%B3%E6%BB%91%E7%A7%BB%E5%8A%A8%E5%B9%B3%E5%9D%87%E7%BA%BF

# Get stock's macd
# input: ema12     pandas data frame column object
#        ema26     pandas data frame column object
# return: pandas data frame object
def get_moving_average_convergence_divergence(ema12, ema26):
    return ema12 - ema26


# Get stock's bollinger_bands
# input: stock_data     pandas data frame object
#        ma21           pandas data frame column object
# return: pandas data frame object
def get_bollinger_bands(stock_data, ma21):
    std20 = pd.Series(stock_data["Close"]).rolling(window=20).std()
    upper_band = ma21 + std20 * 2
    lower_band = ma21 - std20 * 2
    return std20, upper_band, lower_band


# Technical indicators
# (ma7, ma21, 26ema, 12ema, macd, 20sd, upperband, lowerband, momentum)
# Preprocess stock Data to get its technical indicators
# input: stock_data     pandas data frame object
# return: pandas data frame object
def get_technical_indicators(stock_data):
    # 7 and 21 days MA
    stock_data['Ma7'] = get_simple_moving_average(stock_data, 7)
    stock_data['Ma21'] = get_simple_moving_average(stock_data, 21)

    # 26 and 21 days EMA
    stock_data['Ema26'] = get_exp_moving_average(stock_data, 26)
    stock_data['Ema12'] = get_exp_moving_average(stock_data, 12)

    # MACD
    stock_data['MACD'] = get_moving_average_convergence_divergence(stock_data['Ema12'],
                                                                   stock_data['Ema26'])
    # 10 days Momentum
    stock_data['Momentum'] = get_momentum(stock_data, 10)

    # EMA
    stock_data['Ema'] = get_exp_moving_average2(stock_data, 0.5)

    # Bollinger Bands
    stock_data['Std20'], stock_data['UpperBand'], stock_data['LowerBand'] = get_bollinger_bands(stock_data,
                                                                                                stock_data['Ma21'])
    # TODO Other Technical Indicators
    return stock_data
