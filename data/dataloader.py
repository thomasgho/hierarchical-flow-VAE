import pickle
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X): #, y):
        super(Dataset, self).__init__()
        self.X = X
        #self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx]#, self.y[idx]


def dataloader(rootdir, batch_size):

    with open(rootdir, 'rb') as f:
        dataset = pickle.load(f)

    dataset = torch.unsqueeze(torch.as_tensor(dataset), 1)
    dataset = Dataset(dataset)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

    return trainloader, testloader
