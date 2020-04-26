from prediction.model.series_model.lstm import LSTM
from prediction.model.series_model.bi_lstm_attention import BiLSTMAttention
import torch
from torch import optim
from tensorboardX import SummaryWriter
import pandas as pd
import numpy as np
import copy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter("prediction/resources/tensorboards")


def loss_function():
    return torch.nn.MSELoss()


def train(optimizer, model, lstm_data_loader, autoencoder):
    model.train()
    if autoencoder is not None:
        autoencoder.eval()
    train_loss = 0
    for batch_idx, (seq_in, target) in enumerate(lstm_data_loader):
        optimizer.zero_grad()

        seq_in = seq_in.to(device)
        target = target.to(device)

        if autoencoder is not None:
            h = autoencoder.encode(seq_in)
            y = model(h)
        else:
            y = model(seq_in)

        loss = loss_function()(y, target)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(lstm_data_loader.dataset)
    return train_loss

    '''
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(lstm_data_loader.dataset)))

    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #
    # ref: https://blog.csdn.net/qq_39575835/article/details/89160828
    # 1. Log scalar values (scalar summary)
    info = {'loss': train_loss / len(lstm_data_loader.dataset)}

    for tag, value in info.items():
        writer.add_scalars('data/scalar_group', info, epoch)

    # 2. Log values and gradients of the parameters (histogram summary)
    for tag, value in models.named_parameters():
        tag = tag.replace('.', '/')
        writer.add_histogram(tag, value.data.cpu().numpy(), epoch)
        writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch)
    '''


def validate(epoch, model, lstm_data_loader, autoencoder):
    y_cpu = None
    target_cpu = None

    model.eval()
    if autoencoder is not None:
        autoencoder.eval()
    vad_loss = 0
    with torch.no_grad():
        for batch_idx, (seq_in, target) in enumerate(lstm_data_loader):
            seq_in = seq_in.to(device)
            target = target.to(device)

            if autoencoder is not None:
                h = autoencoder.encode(seq_in)
                y = model(h)
            else:
                y = model(seq_in)

            loss = loss_function()(y, target)
            vad_loss += loss.item()

            y_cpu = y.cpu().detach().numpy()
            target_cpu = target.cpu().detach().numpy()

    vad_loss /= len(lstm_data_loader.dataset)

    # ================================================================== #
    #                       Plot Graph                                   #
    # ================================================================== #
    if epoch % 10 == 0:
        dates = pd.date_range('2020-01-01', periods=len(lstm_data_loader.dataset))
        results = np.zeros((len(lstm_data_loader.dataset), 2))
        for i in range(len(lstm_data_loader.dataset)):
            results[i, 0] = y_cpu[i, 0]
            results[i, 1] = target_cpu[i, 0]
        print(results)

    return vad_loss


def train_lstm(epochs, train_lstm_data_loader, vad_lstm_data_loader, state_dict=None, autoencoder=None):
    if autoencoder is not None:
        model = LSTM(input_dim=autoencoder.encode_dim, seq_len=train_lstm_data_loader.dataset.seq_length).to(device)
    else:
        model = LSTM(seq_len=train_lstm_data_loader.dataset.seq_length).to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)
        return model.eval()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_loss = 10000.0
    best_ep = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_loss = train(optimizer, model, train_lstm_data_loader, autoencoder)
        val_loss = validate(epoch, model, vad_lstm_data_loader, autoencoder)
        if val_loss < best_loss:
            best_loss = val_loss
            best_ep = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        print("Best val Loss:", best_loss, "in epoch:", best_ep)
        if epoch - best_ep > 150:
            print('early stop')
            break

    model.load_state_dict(best_model_wts)
    return model.eval()


def test_lstm(model, lstm_data_loader, autoencoder):
    y_cpu = None

    model.eval()
    if autoencoder is not None:
        autoencoder.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (seq_in, target) in enumerate(lstm_data_loader):
            seq_in = seq_in.to(device)
            target = target.to(device)

            if autoencoder is not None:
                h = autoencoder.encode(seq_in)
                y = model(h)
            else:
                y = model(seq_in)

            loss = loss_function()(y, target)
            test_loss += loss.item()

            y_cpu = y.cpu().detach().view(-1).numpy()

    test_loss /= len(lstm_data_loader.dataset)
    return test_loss, pd.DataFrame(y_cpu, columns=['Predicted_Close'])
