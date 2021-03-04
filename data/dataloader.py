import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


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


# define material dataloader
def material_loader(data_loc, batch_size):
    '''
    returns one train set and two validation sets (ratio 60:20:20)
    '''

    # load data
    data = np.load(data_loc, allow_pickle=True)

    # attributes
    id = data[:,0]
    atom_type = data[:,1]
    xrd = data[:,2:3602]
    space_group = (data[:,3602]-1).astype(int)
    band_gap = data[:,3603]
    energy = data[:,3604]
    mag_moment = data[:,3605]
    energy_above_hull = data[:,3606]
    bravais = (data[:,3607]-1)
    targets = data[:,3602:]

    # split train and test data
    X_train, X_test, energy_train, energy_test, bravais_train, bravais_test = train_test_split(xrd, energy, bravais, test_size=0.40, shuffle=True, random_state=9)
    X_val_1, X_val_2, energy_val_1, energy_val_2, bravais_val_1, bravais_val_2 = train_test_split(X_test, energy_test, bravais_test, test_size=0.50, shuffle=True, random_state=9)
    train_dataset = ndarrayDataset(X_train, energy_train, bravais_train)
    val_dataset_1 = ndarrayDataset(X_val_1, energy_val_1, bravais_val_1)
    val_dataset_2 = ndarrayDataset(X_val_2, energy_val_2, bravais_val_2)

    # pytorch dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader_1 = DataLoader(val_dataset_1, batch_size=batch_size)
    val_loader_2 = DataLoader(val_dataset_2, batch_size=batch_size)

    return train_loader, val_loader_1, val_loader_2