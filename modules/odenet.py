import torch
from torch import nn
#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint

class ODEfunc(nn.Module):
    """
    Network architecture for ODENet.
    """
    def __init__(self, dim, dropout=0.0, zero_initialization=True, batch_norm=True):
        super(ODEfunc, self).__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])
        self.dropout = nn.Dropout(p=dropout)
        self.relu =  nn.ReLU()
        self.batch_norm = batch_norm
        if zero_initialization:
            nn.init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            nn.init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)
        if batch_norm:
            self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(dim, eps=1e-3) for _ in range(2)])
        self.nfe = 0    # Number of function evaluations

    def forward(self, t, inputs):
        self.nfe += 1
        temps = inputs
        if self.batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.relu(temps)
        temps = self.linear_layers[0](temps)
        if self.batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.relu(temps)
        temps = self.dropout(temps)
        outputs = self.linear_layers[1](temps)
        return outputs


class ODEblock(nn.Module):
    def __init__(self, odefunc, tol):
        super(ODEblock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.tol = tol

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class ODEnet(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, num_blocks=2, dropout=0.4, gauss_encoder=False, gauss_mix=False, num_gauss = 14, tol=1e-3):
        super().__init__()
        self.initial_layer = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ODEblock(ODEfunc(hidden_dim, dropout=dropout), tol) for _ in range(num_blocks)])
        self.final_layer = nn.Linear(hidden_dim, out_dim)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.encode = gauss_encoder
        self.gaussian_mix = gauss_mix
        self.num_gauss = num_gauss
        # if using ODE as a VAE encoder
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
            temps = self.relu(block(temps))
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
