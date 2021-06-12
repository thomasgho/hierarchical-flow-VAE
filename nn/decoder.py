from nn.mlp import GatedDense
from torchdyn.models import *


class Combiner(nn.Module):
    def __init__(self, dim):
        super(Combiner, self).__init__()
        self.net = nn.Conv1d(dim+dim, dim, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.net(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim):
        super(DecoderBlock, self).__init__()

        self.vec_field = nn.Sequential(
            DataControl(),
            nn.BatchNorm1d(dim+dim),
            nn.SiLU(),
            nn.ConvTranspose1d(dim+dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.ConvTranspose1d(dim, dim, kernel_size=3, padding=1))

        self.neural_ode = NeuralDE(
            self.vec_field,
            sensitivity='adjoint',
            solver='euler')

    def forward(self, x):
        return self.neural_ode(x)


class Decoder(nn.Module):
    def __init__(self, output_dim, feat_dims=[64, 32], z_dims=[15, 15]):
        super(Decoder, self).__init__()

        self.conv_levels = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels = feat_dims[i],
                    out_channels = 1 if feat_dim == feat_dims[-1] else feat_dims[i+1],
                    kernel_size = 6,
                    stride = 2,
                    padding = 2,
                    bias = False),
                DecoderBlock(1 if feat_dim == feat_dims[-1] else feat_dims[i+1]),
                DecoderBlock(1 if feat_dim == feat_dims[-1] else feat_dims[i+1]),
                DecoderBlock(1 if feat_dim == feat_dims[-1] else feat_dims[i+1]))
            for i, feat_dim in enumerate(feat_dims)])

        self.reparamd_levels = nn.ModuleList([
            nn.Sequential(
                GatedDense(z_dim, feat_dim*(output_dim//(2**(len(feat_dims)-i))), activation=nn.SiLU()),
                nn.Unflatten(1, (feat_dim, output_dim//(2**(len(feat_dims)-i)))))
            for i, (feat_dim, z_dim) in enumerate(zip(feat_dims, z_dims))])

        self.comb_levels = nn.ModuleList([
            Combiner(feat_dim) for feat_dim in feat_dims])

    def forward(self, z_list):   # tensor of z points from each level of encoder x=[z1,z2,...]

        x_list = [self.reparamd_levels[lvl](z) for lvl, z in enumerate(reversed(z_list))]

        x = self.conv_levels[0](x_list[0])
        for lvl in range(1, len(x_list)):
            x = self.comb_levels[lvl](x, x_list[lvl])
            x = self.conv_levels[lvl](x)

        return x



def test():
    x = torch.randn((2, 3, 15))
    model = Decoder(output_dim=100, z_dims=[15, 15], feat_dims=[64, 32])
    pred = model(x)
    print(pred.shape)

if __name__ == "__main__":
    test()