from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


# VAE reference code
# ref: https://github.com/pytorch/examples/blob/master/vae/main.py
# ref: https://shenxiaohai.me/2018/10/20/pytorch-tutorial-advanced-02/

# Tutorial of VAE
# ref: https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73

# De-nosing variational auto encoder for input embedding
class VAE(nn.Module):
    def __init__(self, input_dim=31, latent_dim=20, out_dim=31, hidden_dim=400):
        super(VAE, self).__init__()

        # VAE Encoder
        # Layer: (input_dim, n) -> (n,n) -> (n,n) -> (n,latent_dim) [mean]
        #                                         -> (n,latent_dim) [std]
        self.encode_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encode_fc21 = nn.Linear(hidden_dim, latent_dim)
        self.encode_fc22 = nn.Linear(hidden_dim, latent_dim)

        # VAE Decoder
        # Layer: (latent_dim, n) -> (n,n) -> (n,n) -> (n,outdim)
        self.decode_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decode_fc2 = nn.Linear(hidden_dim, out_dim)

    def encode(self, x):
        h1 = F.relu(self.encode_fc1(x))
        return self.encode_fc21(h1), self.encode_fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h1 = F.relu(self.decode_fc1(z))
        return torch.sigmoid(self.decode_fc2(h1))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


