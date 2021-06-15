from nn.encoder import *
from nn.decoder import *
from nn.flow import *



class VAE(nn.Module):
    def __init__(self, input_dim, feat_dims=[32, 64], z_dims=[15, 15], trace_estimator='hutchinson'):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, feat_dims, z_dims)
        self.decoder = Decoder(input_dim, list(reversed(feat_dims)), list(reversed(z_dims)))
        self.flow = nn.ModuleList([NeuralDE(Flow(z, trace_estimator=trace_estimator), solver='dopri5', sensitivity='adjoint') for z in z_dims])
        self.augment = Augmenter(augment_idx=1, augment_dims=1)

    def forward(self, x):
        z_levels, mu_levels, var_levels = self.encoder(x)   # z_levels = [z1,z2,...]

        z_flows = []
        trJs = []
        for lvl, flow in enumerate(self.flow):
            ztrJ = flow(self.augment(z_levels[lvl]))
            z, trJ = ztrJ[:, 1:], ztrJ[:, 0]
            z_flows.append(z)
            trJs.append(trJ)

        ###################
        # SE(3) TRANSFORM #
        ###################

        x_recon = self.decoder(z_flows)
        return x_recon, mu_levels, var_levels, z_flows, trJs


def test():
    x = torch.randn((3,1,100))
    model = VAE(input_dim=100, z_dims=[15, 15], feat_dims=[32, 64])
    x_recon, mu_levels, var_levels, z_flows, trJs = model(x)
    print(x_recon.shape)

if __name__ == "__main__":
    test()








