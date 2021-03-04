import torch
import torch.utils.data
from torch.utils.data import Dataset


# define auxiliary dataset class
class ndarrayDataset(Dataset):
    """simple dataset"""

    def __init__(self, X, y_structure, y_energy):
        super(ndarrayDataset, self).__init__()
        self.X = torch.from_numpy(X).float()
        self.y_structure = torch.from_numpy(y_structure).float()
        self.y_energy = torch.from_numpy(y_energy).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y_structure[idx], self.y_energy[idx]
