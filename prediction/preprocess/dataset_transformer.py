import torch
import torch.utils.data as data


class TransformerDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, seqs_in, targets, seq_length=7):
        self.seqs_in = seqs_in
        self.targets = targets
        self.seq_length = seq_length

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        if self.targets is not None:
            return torch.FloatTensor(self.seqs_in.iloc[index: index + self.seq_length].to_numpy()), \
                   torch.FloatTensor([self.seqs_in.iloc[index + self.seq_length - 1]]), \
                   torch.FloatTensor(self.targets.iloc[index + self.seq_length])
        return torch.FloatTensor(self.seqs_in.iloc[index: index + self.seq_length].to_numpy())

    def __len__(self):
        return self.seqs_in.shape[0] - self.seq_length


# Get Transformer DataLoaders
def get_transformer_data_loader(seqs_in, targets, test=False, batch_size=64):
    transformer_dataset = TransformerDataset(seqs_in, targets)
    data_loader = torch.utils.data.DataLoader(dataset=transformer_dataset,
                                              batch_size=batch_size,
                                              shuffle=not test)
    return data_loader
