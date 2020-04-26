from prediction.train.train_lstm import train_lstm
from prediction.train.train_transformer import train_transformer
from prediction.train.train_lstm import test_lstm
from prediction.preprocess.dataset_lstm import get_lstm_data_loader
from prediction.preprocess.dataset_transformer import get_transformer_data_loader
from prediction.preprocess.preprocess import fetch_and_preprocess_mass_datasets
from prediction.preprocess.preprocess import fetch_and_preprocess_one_datasets
import torch
import os
import pandas as pd


def train_model(stock_code, train_input, train_labels, test_input, test_labels, ep=1500, use_cache=True):
    model_path = 'prediction/resources/models/' + 'stock_' + stock_code + '_model.pkl'

    train_lstm_data_loader = get_lstm_data_loader(train_input, train_labels)
    vad_lstm_data_loader = get_lstm_data_loader(test_input, test_labels, test=True, batch_size=len(test_input))

    if os.path.isfile(model_path) and use_cache:
        model = train_lstm(ep, train_lstm_data_loader, vad_lstm_data_loader,
                           state_dict=torch.load(model_path))
        return model

    model = train_lstm(ep, train_lstm_data_loader, vad_lstm_data_loader)
    torch.save(model.state_dict(), 'prediction/resources/models/' + 'stock_' + stock_code + '_model.pkl')
    return model


#train_inputs, train_labels, test_inputs, test_labels = fetch_and_preprocess_one_datasets()
#train_model(stock_code="0005.HK", train_input=train_inputs, train_labels=train_labels, test_input=test_inputs, test_labels=test_labels, use_cache=False)


def train_models(ep=1500):
    mass_train_inputs, mass_train_labels, mass_vad_inputs, mass_vad_labels, mass_test_inputs, mass_test_labels, fail_preprocess_stocks, succ_stocks_codes = fetch_and_preprocess_mass_datasets()
    for idx, stock_code in enumerate(succ_stocks_codes):
        print("training " + stock_code)
        print("current_index: " + str(idx))
        model = train_model(stock_code, mass_train_inputs[idx], mass_train_labels[idx], mass_vad_inputs[idx],
                            mass_vad_labels[idx], ep=ep, use_cache=True)

        test_lstm_data_loader = get_lstm_data_loader(mass_test_inputs[idx], mass_test_labels[idx], test=True,
                                                     batch_size=len(mass_test_inputs[idx]))
        test_loss, result_labels = test_lstm(model, test_lstm_data_loader, autoencoder=None)

        mass_test_labels[idx] = mass_test_labels[idx].iloc[test_lstm_data_loader.dataset.seq_length:]

        mass_test_labels[idx].reset_index(inplace=True)
        df = pd.concat([result_labels['Predicted_Close'], mass_test_labels[idx]], axis=1)
        df.set_index('Date')
        df.to_csv('prediction/resources/test/' + 'predicted_stock_' + stock_code + '.csv')


train_models(1500)

'''
# AE
train_ae_data_loader = get_ae_data_loader(train_input)
test_ae_data_loader = get_ae_data_loader(test_input, test=True, batch_size=len(test_input))
models = train_ae(1500, train_ae_data_loader, test_ae_data_loader)
'''

'''
# LSTM
train_lstm_data_loader = get_lstm_data_loader(train_input, train_labels)
test_lstm_data_loader = get_lstm_data_loader(test_input, test_labels, test=True, batch_size=len(test_input))
train_lstm(1500, train_lstm_data_loader, test_lstm_data_loader, autoencoder=None)
'''

'''
# Transformer
train_transformer_data_loader = get_transformer_data_loader(train_input, train_labels)
test_transformer_data_loader = get_transformer_data_loader(test_input, test_labels, test=True, batch_size=len(test_input))
train_lstm(1500, train_transformer_data_loader, test_transformer_data_loader)
'''
