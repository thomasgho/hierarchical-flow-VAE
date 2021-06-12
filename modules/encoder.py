from nn.mlp import GatedDense
from torchdyn.models import *


class EncoderBlock(nn.Module):
    def __init__(self, dim):
        super(EncoderBlock, self).__init__()

        self.vec_field = nn.Sequential(
            DataControl(),
            nn.BatchNorm1d(dim+dim),
            nn.SiLU(),
            nn.Conv1d(dim+dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1))

        self.neural_ode = NeuralDE(
            self.vec_field,
            sensitivity='adjoint',
            solver='euler')

    def forward(self, x):
        return self.neural_ode(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, feat_dims=[32, 64], z_dims=[15, 15]):
        super(Encoder, self).__init__()

        self.conv_levels = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels = 1 if feat_dim == feat_dims[0] else feat_dims[i-1],
                    out_channels = feat_dims[i],
                    kernel_size = 6,
                    stride = 2,
                    padding = 2,
                    bias = False),
                EncoderBlock(feat_dims[i]),
                EncoderBlock(feat_dims[i]),
                EncoderBlock(feat_dims[i]))
            for i, feat_dim in enumerate(feat_dims)])

        self.mu_levels = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                GatedDense(feat_dim*(input_dim//(2**(i+1))), z_dim, activation=nn.SiLU()))
            for i, (feat_dim, z_dim) in enumerate(zip(feat_dims, z_dims))])

        self.var_levels = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                GatedDense(feat_dim*(input_dim//(2**(i+1))), z_dim, activation=nn.SiLU()))
            for i, (feat_dim, z_dim) in enumerate(zip(feat_dims, z_dims))])

    def forward(self, x):
        mu_levels, var_levels, z_levels = [], [], []
        for conv_net, mu_net, var_net in zip(self.conv_levels, self.mu_levels, self.var_levels):
            x = conv_net(x)
            mu, var = mu_net(x), torch.exp(var_net(x))
            z = torch.distributions.Normal(mu, var.pow(0.5)).sample()
            mu_levels.append(mu)
            var_levels.append(var)
            z_levels.append(z)
        return torch.stack(z_levels), torch.stack(mu_levels), torch.stack(var_levels)  # z_levels = [z1,z2,...]


def test():
    x = torch.randn((3,1,100))
    model = Encoder(input_dim=100, z_dims=[15, 15], feat_dims=[32, 64])
    z_list, mu_list, var_list = model(x)
    print(z_list.shape)

if __name__ == "__main__":
    test()