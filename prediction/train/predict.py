import pandas as pd
import os


def get_predict_test_set(date="2020-01-22", stock_code="0007.hk"):
    predict_stock_path = '../resources/test/' + 'predicted_stock_' + stock_code + '.csv'
    if os.path.isfile(predict_stock_path):
        df = pd.read_csv(predict_stock_path, index_col="Date")

        if date in df.index:
            index = df.index.get_loc(date)
            if index is not 0:
                profit = df.iloc[index]['Predicted_Close'] - df.iloc[index - 1]['Predicted_Close']
                return profit
    return None


profit = get_predict_test_set()
