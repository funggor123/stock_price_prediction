from prediction.model.ae_model.vae import VAE
import torch
import torch.nn.functional as F
from torch import optim

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Log interval
log_interval = 10

torch.manual_seed(1)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(epoch, optimizer, model, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch, model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def predict_vae(model, predict_loader):
    recon_batch = None
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(predict_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
    return recon_batch.cpu().detach().numpy()


def train_vae(epochs, train_vae_data_loader, test_vae_data_loader):
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        train(epoch, optimizer, model, train_vae_data_loader)
        test(epoch, model, test_vae_data_loader)
    return model


