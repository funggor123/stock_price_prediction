import torch
import torch.utils.data as data


class AEDataset(data.Dataset):

    def __init__(self, seqs_in):
        self.seqs_in = seqs_in

    def __getitem__(self, index):
        return torch.FloatTensor(self.seqs_in.iloc[index].to_numpy())

    def __len__(self):
        return self.seqs_in.shape[0]


# Get AE DataLoaders
def get_ae_data_loader(df, test=False, batch_size=128):
    vae_dataset = AEDataset(df)
    data_loader = torch.utils.data.DataLoader(dataset=vae_dataset,
                                              batch_size=batch_size,
                                              shuffle=not test)
    return data_loader
