from prediction.model.ae_model.ae import AE
import torch
from torch import optim
import copy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)


def loss_function():
    return torch.nn.MSELoss(reduction='sum')


def train(optimizer, model, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, data_true in enumerate(train_loader):
        optimizer.zero_grad()

        data_true = data_true.to(device)
        data_pred = model(data_true)

        loss = loss_function()(data_pred, data_true)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)
    return train_loss


def test(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data_true in enumerate(test_loader):
            data_true = data_true.to(device)
            data_pred = model(data_true)

            loss = loss_function()(data_pred, data_true)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    return test_loss


def predict_ae(model, predict_loader):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(predict_loader):
            data = data.to(device)
            latent = model(data)
    return latent.cpu().detach().numpy()


def train_ae(epochs, train_ae_data_loader, test_ae_data_loader, model=None):
    if model is None:
        model = AE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_loss = 10000.0
    best_ep = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_loss = train(optimizer, model, train_ae_data_loader)
        test_loss = test(model, test_ae_data_loader)
        if test_loss < best_loss:
            best_loss = test_loss
            best_ep = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f'Epoch {epoch}: train loss {train_loss} val loss {test_loss}')
        print("Best val Loss:", best_loss, "in epoch:", best_ep)

    model.load_state_dict(best_model_wts)
    return model.eval()

