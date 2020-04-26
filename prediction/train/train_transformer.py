from prediction.model.series_model.transformer import TimeSeriesTransformer
import torch
from torch import optim
from tensorboardX import SummaryWriter
import pandas as pd
import numpy as np
import copy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Log interval
log_interval = 10

writer = SummaryWriter("../resources/tensorboards")


def loss_function():
    return torch.nn.MSELoss()


def get_batch(source):
    return source.permute(1, 0, 2)


def train(epoch, optimizer, model, transformer_data_loader):
    model.train()
    train_loss = 0
    for batch_idx, (seq_in, decode_in, target) in enumerate(transformer_data_loader):
        seq_in = get_batch(seq_in).to(device)
        decode_in = get_batch(decode_in).to(device)
        target = target.to(device)

        optimizer.zero_grad()
        y = model(seq_in, decode_in)
        loss = loss_function()(y, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        train_loss += loss.item()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(transformer_data_loader.dataset)))

    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #
    # ref: https://blog.csdn.net/qq_39575835/article/details/89160828
    # 1. Log scalar values (scalar summary)
    info = {'loss': train_loss / len(transformer_data_loader.dataset)}

    for tag, value in info.items():
        writer.add_scalars('data/scalar_group', info, epoch)


def validate(epoch, model, transformer_data_loader):
    y_cpu = None
    target_cpu = None

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (seq_in, decode_in, target) in enumerate(transformer_data_loader):
            seq_in = get_batch(seq_in).to(device)
            decode_in = get_batch(decode_in).to(device)
            target = target.to(device)

            y = model(seq_in, decode_in)
            loss = loss_function()(y, target)

            y_cpu = y.cpu().detach().numpy()
            target_cpu = target.cpu().detach().numpy()

            test_loss += loss.item()

    test_loss /= len(transformer_data_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    # ================================================================== #
    #                       Plot Graph                                   #
    # ================================================================== #
    if epoch % log_interval == 0:
        dates = pd.date_range('2020-01-01', periods=len(transformer_data_loader.dataset))
        results = np.zeros((len(transformer_data_loader.dataset), 2))
        for i in range(len(transformer_data_loader.dataset)):
            results[i, 0] = y_cpu[i, 0]
            results[i, 1] = target_cpu[i, 0]
        print(results)

    return test_loss

'''
epochs, train_lstm_data_loader, vad_lstm_data_loader, state_dict=None, autoencoder=None
'''


def train_transformer(epochs, train_transformer_data_loader, vad_lstm_data_loader, state_dict=None):
    model = TimeSeriesTransformer(seq_length=len(train_transformer_data_loader.dataset)).to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)
        return model.eval()

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.99)

    best_loss = 10000.0
    best_ep = 0
    best_model_wts = copy.deepcopy(model.state_dict())


    for epoch in range(1, epochs + 1):
        train_loss = train(epoch=epoch, optimizer=optimizer, model=model, transformer_data_loader=train_transformer_data_loader)
        val_loss = validate(epoch, model, vad_lstm_data_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            best_ep = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
        scheduler.step(epoch)
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        print("Best val Loss:", best_loss, "in epoch:", best_ep)

    model.load_state_dict(best_model_wts)
    return model.eval()


