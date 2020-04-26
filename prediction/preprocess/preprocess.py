from prediction.cron.cron_correlated_assets import CURRENCY_LIST
from prediction.cron.cron_correlated_assets import COMPOSITE_INDICES_LIST
from prediction.preprocess.technical_analysis import get_technical_indicators
from prediction.preprocess.trend_analysis import fourier_transform
from prediction.cron.cron_correlated_assets import fetch_currencies
from prediction.cron.cron_correlated_assets import fetch_composite_indices
from prediction.cron.cron_stock import fetch_stock
from prediction.cron.cron_stock import fetch_stocks
import os.path
import pandas as pd

from sklearn import preprocessing

min_max_column_list = [
    'Open', 'High', 'Low', 'Close', 'Ma7', 'Ma21', 'Ema26', 'Ema12', 'MACD',
    'Momentum', 'Ema', 'Std20', 'UpperBand', 'LowerBand', 'FT_3', 'FT_6', 'FT_9',
    '^IXIC_Close', '^GSPC_Close', '^DJI_Close', '^HSI_Close', '^N225_Close',
    '^BSESN_Close', '^FTSE_Close', '^FCHI_Close', '^GDAXI_Close',
    'USDHKD=X_Close', 'EURHKD=X_Close', 'GBPHKD=X_Close', 'AUDHKD=X_Close',
    'CADHKD=X_Close', 'SGDHKD=X_Close', 'MYRHKD=X_Close', 'CNYHKD=X_Close'
]


# Transform Features
def transform_features(df):
    # Normalize Data
    scaler = preprocessing.MinMaxScaler()
    for col in min_max_column_list:
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

    # extract_datetime(df)


# Select features
# input:    stock_data                    pandas data frame object
#           composite_indices_df_list     industries of pandas data frame object
#           currencies_df_list            industries of pandas data frame object
# return:   pandas data frame object
def select_features(stock_data, composite_indices_df_list, currencies_df_list):
    df = stock_data

    # Drop unused features
    df.drop(columns=['Volume', 'Adj Close'], axis=1, inplace=True)

    # Get Only Closed Price in Composite indices
    for i, composite_indices in enumerate(COMPOSITE_INDICES_LIST):
        try:
            df = pd.merge(df, composite_indices_df_list[i][[composite_indices + "_" + "Close"]], how='left',
                          left_index=True, right_index=True)
        except Exception as e:
            return None

    # Get Only Closed Price in Currencies
    for i, currencies in enumerate(CURRENCY_LIST):
        df = pd.merge(df, currencies_df_list[i][[currencies + "_" + "Close"]], how='left',
                      left_index=True, right_index=True)

    return df


MIN_SAMPLES = 640


def select_stock(stock_data):
    if stock_data['Close'].count() > MIN_SAMPLES:
        return True
    return False


# Preprocess features
# input:    stock_data                    pandas data frame object
#           composite_indices_df_list     industries of pandas data frame object
#           currencies_df_list            industries of pandas data frame object
# return:   pandas data frame object
def preprocess(stock_data, composite_indices_df_list, currencies_df_list):
    # Drop Not Suitable Stock
    if select_stock(stock_data) is not True:
        return None, None

    # Null Analysis for stock data
    stock_data_null_analysis(stock_data)

    # Get Results from Technical Analysis
    stock_data = get_technical_indicators(stock_data)
    stock_data.dropna(how='any', inplace=True)

    # Get Results From Trend Analysis
    fourier_transform(stock_data)

    # Feature Selection
    data_set = select_features(stock_data, composite_indices_df_list, currencies_df_list)
    if data_set is None:
        return None, None

    # null_analysis for composite
    null_analysis(data_set,
                  columns=['^IXIC_Close', '^GSPC_Close', '^DJI_Close', '^HSI_Close', '^N225_Close',
                           '^BSESN_Close', '^FTSE_Close', '^FCHI_Close', '^GDAXI_Close'])
    # null_analysis for curr
    null_analysis(data_set,
                  columns=['^BSESN_Close', '^FTSE_Close', '^FCHI_Close', '^GDAXI_Close',
                           'USDHKD=X_Close', 'EURHKD=X_Close', 'GBPHKD=X_Close', 'AUDHKD=X_Close',
                           'CADHKD=X_Close', 'SGDHKD=X_Close', 'MYRHKD=X_Close', 'CNYHKD=X_Close'])

    # Predicted Targets
    labels = stock_data['Close']

    # Feature Transformation
    # All input features must be transformed to 0 - 1 !
    transform_features(data_set)

    get_data_set_summary(data_set)
    return data_set, labels


# Interpolation
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.interpolate.html
def null_analysis(df, columns=None):
    if columns is not None:
        for column in columns:
            df[column] = df[column].interpolate(method='linear', limit=100)
            # TODO
            df[column].fillna(0, inplace=True)


# Handle Invalid / Empty Records
# fill with zero values for NaN records
# ref: https://stackoverflow.com/questions/52570199/multivariate-lstm-with-missing-values
def stock_data_null_analysis(df, columns=None):
    # df.fillna(0, inplace=True)
    # print(df.isnull().sum())
    return None


def extract_datetime(df):
    datetime = pd.to_datetime(df.index)
    df['Month'] = datetime.month
    df['Day'] = datetime.day
    return df


# Split dataset
def split_dataset(dataset):
    train_dataset = dataset.loc['2004-01-01':'2019-01-01']
    vad_dataset = dataset.loc['2019-01-01':'2020-01-01']
    test_dataset = dataset.loc['2020-01-01':'2020-04-01']
    return train_dataset, vad_dataset, test_dataset


MIN_DATA_SET_SIZE = 64


# Get dataset
def get_datasets(stock_data, stock_code, composite_indices_df_list, currencies_df_list, use_cache=False):
    train_input_path = 'prediction/resources/train/train_data_' + stock_code + '_.csv'
    train_labels_path = 'prediction/resources/train/train_data_label_' + stock_code + '_.csv'

    vad_input_path = 'prediction/resources/vad/vad_data_' + stock_code + '_.csv'
    vad_labels_path = 'prediction/resources/vad/vad_data_label_' + stock_code + '_.csv'

    test_input_path = 'prediction/resources/test/test_data_' + stock_code + '_.csv'
    test_labels_path = 'prediction/resources/test/test_data_label_' + stock_code + '_.csv'

    if use_cache and os.path.isfile(train_input_path) and os.path.isfile(train_labels_path) and os.path.isfile(
            vad_input_path) and os.path.isfile(vad_labels_path) and os.path.isfile(
        test_input_path) and os.path.isfile(test_labels_path):
        train_input = pd.read_csv(train_input_path, index_col=0)
        train_labels = pd.read_csv(train_labels_path, index_col=0)

        vad_input = pd.read_csv(vad_input_path, index_col=0)
        vad_labels = pd.read_csv(vad_labels_path, index_col=0)

        test_input = pd.read_csv(test_input_path, index_col=0)
        test_labels = pd.read_csv(test_labels_path, index_col=0)
        return train_input, train_labels, vad_input, vad_labels, test_input, test_labels

    dataset, labels = preprocess(stock_data, composite_indices_df_list, currencies_df_list)
    if dataset is None:
        return None, None, None, None, None, None

    train_input, vad_input, test_input = split_dataset(dataset)
    train_labels, vad_labels, test_labels = split_dataset(labels)

    # Cache training data
    train_input.to_csv(train_input_path)
    train_labels.to_csv(train_labels_path)

    vad_input.to_csv(vad_input_path)
    vad_labels.to_csv(vad_labels_path)

    test_input.to_csv(test_input_path)
    test_labels.to_csv(test_labels_path)

    if train_input['Close'].count() < MIN_DATA_SET_SIZE:
        return None, None, None, None, None, None

    if vad_input['Close'].count() < MIN_DATA_SET_SIZE:
        return None, None, None, None, None, None

    if test_input['Close'].count() < MIN_DATA_SET_SIZE:
        return None, None, None, None, None, None

    return train_input, train_labels, vad_input, vad_labels, test_input, test_labels


def get_mass_datasets(stocks_data, stocks_code, composite_indices_df_list, currencies_df_list, use_cache=False):
    mass_train_inputs, mass_train_labels, mass_vad_inputs, mass_vad_labels, mass_test_inputs, mass_test_labels, fail_stocks_code, succ_stocks_code = [], [], [], [], [], [], [], []
    for idx, stock_data in enumerate(stocks_data):
        print("preprocessing  " + stocks_code[idx])
        print('current_index: ' + str(idx))
        train_inputs, train_labels, vad_inputs, vad_labels, test_inputs, test_labels = get_datasets(stock_data,
                                                                                                    stocks_code[idx],
                                                                                                    composite_indices_df_list,
                                                                                                    currencies_df_list,
                                                                                                    use_cache)
        if train_inputs is not None:
            mass_train_inputs.append(train_inputs)
            mass_train_labels.append(train_labels)
            mass_vad_inputs.append(vad_inputs)
            mass_vad_labels.append(vad_labels)
            mass_test_inputs.append(test_inputs)
            mass_test_labels.append(test_labels)
            succ_stocks_code.append(stocks_code[idx])
        else:
            fail_stocks_code.append(stocks_code[idx])
    return mass_train_inputs, mass_train_labels, mass_vad_inputs, mass_vad_labels, mass_test_inputs, mass_test_labels, fail_stocks_code, succ_stocks_code


# Get Dataset summary
def get_data_set_summary(dataset):
    print("data_set summary")
    print(dataset)
    # print(dataset.columns)
    # print(dataset.shape)


def fetch_and_preprocess_mass_datasets():
    stocks_data, stocks_code, fail_cron_stocks = fetch_stocks(use_cache=True)
    print("num of fail fetch stock: " + str(len(fail_cron_stocks)))
    mass_train_inputs, mass_train_labels, mass_vad_inputs, mass_vad_labels, mass_test_inputs, mass_test_labels, fail_preprocess_stocks, succ_stocks_codes = get_mass_datasets(
        stocks_data,
        stocks_code,
        fetch_composite_indices(
            use_cache=True),
        fetch_currencies(
            use_cache=True),
        use_cache=True)
    print("num of fail preprocess stock: " + str(len(fail_preprocess_stocks)))
    print("num of succ preprocess stock: " + str(len(succ_stocks_codes)))
    return mass_train_inputs, mass_train_labels, mass_vad_inputs, mass_vad_labels, mass_test_inputs, mass_test_labels, fail_preprocess_stocks, succ_stocks_codes


def fetch_and_preprocess_one_datasets(stock_code='0005.HK', start='2003-12-01', end='2020-04-01'):
    train_inputs, train_labels, vad_input, vad_labels, test_input, test_labels = get_datasets(fetch_stock(
        cache_stock_data_path='prediction/resources/raw/stock_data_' + stock_code + '_' + start + '_' + end + '.csv',
        stock_code=stock_code),
        stock_code,
        fetch_composite_indices(
            use_cache=True),
        fetch_currencies(
            use_cache=True), use_cache=True)
    return train_inputs, train_labels, vad_input, vad_labels

