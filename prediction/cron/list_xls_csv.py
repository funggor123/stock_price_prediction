import pandas as pd
import glob
import os


def get_stock_list(stock_list_path, cache_stock_list_path='prediction/resources/industries.csv'):
    if cache_stock_list_path is not None:
        if os.path.isfile(cache_stock_list_path):
            df = pd.read_csv(cache_stock_list_path)
            return df

    path = stock_list_path + "*.csv"
    data = {'Stock_Name': [],
            'Stock_Code': [],
            'Stock_Type': []
            }
    for fname in glob.glob(path):
        df = pd.read_csv(fname)
        for index, row in df.iterrows():
            if index % 2 == 0:
                data['Stock_Name'].append(row['Name/Symbol'])
            if index % 2 == 1:
                data['Stock_Code'].append(row['Name/Symbol'][1:])
                data['Stock_Type'].append(os.path.splitext(os.path.basename(fname))[0])

    df = pd.DataFrame(data, columns=['Stock_Name', 'Stock_Code', 'Stock_Type'], index=None)
    df.to_csv('prediction/resources/industries.csv')
    return df


def get_industries_list(stock_list_path='prediction/resources/industries/'):
    path = stock_list_path + "*.csv"
    stock_types = []
    for fname in glob.glob(path):
        stock_types.append(os.path.splitext(os.path.basename(fname))[0])
    return stock_types


def get_stocks_code_from_stock_list(stock_type, stock_list_csv_path='prediction/resources/industries.csv'):
    df = pd.read_csv(stock_list_csv_path)
    df = df.loc[df['Stock_Type'] == stock_type]
    return df['Stock_Code']


# print(get_stocks_code_from_ind_list("Printing, Publishing & Packaging"))
# get_stock_list('../resources/industries/', cache_stock_list_path='../resources/industries.csv')
