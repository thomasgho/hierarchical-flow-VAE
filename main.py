import argparse
from data.dataloader import dataloader
from nn.model import VAE
import torch
import torch.nn as nn



def loss_fn(X, X_recon, mu_levels, var_levels, trJs):

    kld = 0
    for mu, var in zip(mu_levels, var_levels):
        kld += -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)

    mse = nn.MSELoss(reduction='mean')(X_recon, X)

    mean_trJ = 0
    for trJ in trJs:
        mean_trJ += trJ.mean()

    return (0.1 * kld) + mse - mean_trJ



def train(args):

    # load data
    trainloader, testloader = dataloader(args.path, batch_size=args.batch_size)
    dataloaders = {'train': trainloader, 'val': testloader}

    # initialise CUDA
    if args.cuda:
        device = torch.device("cuda")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type(torch.FloatTensor)

    # define model
    model = VAE(args.in_dim, feat_dims=args.feat_dims, z_dims=args.z_dims, trace_estimator=args.trace).to(device)

    # train loop
    for epoch in range(1, args.epochs + 1):

        # split epochs into training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.beta)
                model.train(True)
            else:
                model.train(False)

            # mini batch training
            running_loss = 0
            for batch, X in enumerate(dataloaders[phase]):
                optimizer.zero_grad()

                # send tensors to CUDA
                X = X.float()
                X = X.to(device)

                # forward propagation
                X_recon, mu_levels, var_levels, _, trJs = model(X)
                loss = loss_fn(X, X_recon, mu_levels, var_levels, trJs)
                running_loss += loss.item()

                # optimization step only in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # print progress
                if batch % args.interval == args.interval - 1:
                    print('epoch: {}, [{}/{} ({:.0f}%)], {} loss: {:.3f}'.format(
                        epoch,
                        batch * args.batch_size,
                        len(trainloader.dataset) if phase == 'train' else len(testloader.dataset),
                        100. * batch / len(trainloader) if phase == 'train' else 100. * batch / len(testloader),
                        phase,
                        running_loss / args.interval))
                    running_loss = 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="str: path directory of data", type=str, default='/home/taymaz/Documents/materialVAE/data/xrd.txt')
    parser.add_argument("--cuda", help="bool: use CUDA acceleration", default=True, action='store_false')
    parser.add_argument("--batch_size", help="int: train batch size", type=int, default=64)
    parser.add_argument("--in_dim", help="int: length of 1d input data", type=list, default=1000)
    parser.add_argument("--feat_dims", help="list: dimensionality of channels in VAE architecture e.g. [32, 64, 128]", type=list, default=[32, 64, 128])
    parser.add_argument("--z_dims", help="list: dimensionality of latent variables at each VAE hierarchy e.g. [15, 15, 15]", type=list, default=[15, 15, 15])
    parser.add_argument("--trace", help="str: trace estimator to use, either 'hutchinson' or 'autograd'", type=str, default='hutchinson')
    parser.add_argument("--epochs", help="int: train epochs", type=int, default=100)
    parser.add_argument("--lr", help="int: ADAM learning rate", type=int, default=1e-5)
    parser.add_argument("--beta", help="int: L2 regularization", type=int, default=0)
    parser.add_argument("--interval", help="int: interval to print batch loss", type=int, default=5)
    args = parser.parse_args()

    train(args)
    
    
