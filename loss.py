import torch
from torch import nn


def vae_loss(recon_x, x, mu, logvar):
    MSE = nn.MSELoss(reduction='mean')(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return 0.1 * MSE + 0.01 * KLD, MSE, KLD

def pred_loss_CE(tar, tar_pred):
    loss = nn.CrossEntropyLoss()
    CE_tar = loss(tar_pred,tar.squeeze())
    return CE_tar

def pred_loss_MSE(tar, tar_pred):
    MSE = nn.MSELoss(reduction='mean')(tar_pred, tar)
    return MSE

def num_matches(tar, tar_pred):
    num = torch.sum(torch.argmax(tar_pred, dim=1).view(-1,1)==tar)
    return num