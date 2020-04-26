from __future__ import print_function
from torch import nn, optim


# Stack AutoEncoder ref: https://medium.com/%E5%BC%B1%E5%BC%B1%E9%96%8B%E7%99%BC%E5%A5%B3%E5%AD%90-%E5%9C%A8%E6%9D%B1
# %E4%BA%AC%E7%9A%84%E9%96%8B%E7%99%BC%E8%80%85%E4%BA%BA%E7%94%9F/autoencoder-%E6%88%91%E5%B0%8D%E4%B8%8D%E8%B5%B7%E4
# %BD%A0%E4%B9%8B-%E9%87%8D%E6%96%B0%E8%AA%8D%E8%AD%98autoencoder-%E7%AC%AC%E4%B8%80%E7%AF%87-d970d1ad9971
class AE(nn.Module):
    def __init__(self, input_dim=34):
        super(AE, self).__init__()
        self.encode_dim = input_dim // 2 // 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 2 // 2),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(input_dim // 2 // 2, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        return x