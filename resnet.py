import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    A general-purpose residual block for 1-dim inputs.

    """

    def __init__(self, dim, dropout=0.4, zero_initialization=True, batch_norm=True):
        super().__init__()
        self.batch_norm = batch_norm
        self.linear_layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        if zero_initialization:
            torch.nn.init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            torch.nn.init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)
        if batch_norm:
            self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(dim, eps=1e-3) for _ in range(2)])

    def forward(self, inputs):
        temps = inputs
        if self.batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.relu(temps)
        temps = self.linear_layers[0](temps)
        if self.batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.relu(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        return inputs + temps


class ResidualNet(nn.Module):
    """
    A general-purpose residual network for 1-dim inputs.
    Option to be used as a Gaussian encoder network (or mixture of Gaussian encoder network)

    """

    def __init__(self, in_dim, out_dim, hidden_dim, num_blocks=2, dropout=0.4, batch_norm=True, gauss_encoder=False,
                 gauss_mix=False, num_gauss=14):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.initial_layer = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(dim=hidden_dim, dropout=dropout, batch_norm=batch_norm) for _ in range(num_blocks)])
        self.final_layer = nn.Linear(hidden_dim, out_dim)
        self.softmax = nn.Softmax(dim=0)
        self.encode = gauss_encoder
        self.gaussian_mix = gauss_mix
        self.num_gauss = num_gauss
        # if using ResNet as a VAE encoder
        if self.encode and not self.gaussian_mix:
            self.final_layer_loc = nn.Linear(hidden_dim, out_dim)
            self.final_layer_scale = nn.Linear(hidden_dim, out_dim)
        if self.encode and self.gaussian_mix:
            self.final_layers_loc = nn.ModuleList([nn.Linear(hidden_dim, out_dim) for _ in range(self.num_gauss)])
            self.final_layers_scale = nn.ModuleList([nn.Linear(hidden_dim, out_dim) for _ in range(self.num_gauss)])
            self.final_layers_weight = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(self.num_gauss)])

    def forward(self, inputs):
        temps = self.initial_layer(inputs)
        for block in self.blocks:
            temps = block(temps)
        if self.encode and not self.gaussian_mix:
            mu = self.final_layer_loc(temps)
            logvar = self.final_layer_scale(temps)
            return mu, logvar
        elif self.encode and self.gaussian_mix:
            mus = torch.stack([self.final_layers_loc[n](temps) for n in range(self.num_gauss)])
            logvars = torch.stack([self.final_layers_scale[n](temps) for n in range(self.num_gauss)])
            weights = torch.stack([self.final_layers_weight[n](temps) for n in range(self.num_gauss)]).squeeze()
            return mus, logvars, self.softmax(weights)
        else:
            outputs = self.final_layer(temps)
            return outputs
