from torch import nn
import pyro
import pyro.distributions as dist
from pyro.distributions.transforms import spline_autoregressive, conditional_spline_autoregressive
from odenet import *
from resnet import *


class FlowVAE(nn.Module):
    """
    Conditional (Autoregressive) Spline Flow based VAE with ResNet/neuralODE encoder/decoder.
    Two hierachies: structural and energetic.
    Option to use a mixture of Gaussians as prior.

    """

    def __init__(self, input_dim, hidden_dim, latent_dim, num_blocks, num_flows, dropout, gauss_mix=False, num_gauss=14, network='resnet'):
        super(FlowVAE, self).__init__()

        self.gaussian_mixture_prior = gauss_mix
        if network == 'resnet':
            self.encoder = ResidualNet(input_dim, latent_dim, hidden_dim, num_blocks=num_blocks, dropout=dropout, gauss_encoder=True, gauss_mix=gauss_mix,
                                       num_gauss=num_gauss)
            self.decoder = ResidualNet(latent_dim, input_dim, hidden_dim, num_blocks=num_blocks, dropout=dropout)
        if network == 'odenet':
            self.encoder = ODEnet(input_dim, latent_dim, hidden_dim, num_blocks=num_blocks, dropout=dropout, gauss_encoder=True, gauss_mix=gauss_mix,
                                  num_gauss=num_gauss, tol=tol)
            self.decoder = ODEnet(latent_dim, input_dim, hidden_dim, num_blocks=num_blocks, dropout=dropout)
        self.flow_structural = [conditional_spline_autoregressive(latent_dim, context_dim=1) for _ in range(num_flows)]
        self.flow_energetic = [conditional_spline_autoregressive(latent_dim, context_dim=1) for _ in range(num_flows)]
        self.flow_modules = nn.ModuleList(self.flow_structural + self.flow_energetic)

    def encode(self, x):
        if self.gaussian_mixture_prior:
            mus, logvars, weights = self.encoder(x)
            return mus, logvars, weights
        else:
            mu, logvar = self.encoder(x)
            return mu, logvar

    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, x, context_structure, context_energy):
        if self.gaussian_mixture_prior:
            mu, logvar, weights = self.encode(x)
            mixture = dist.Categorical(weights.permute(1, 0))
            component = dist.Independent(dist.Normal(mu.permute(1, 0, 2), logvar.permute(1, 0, 2)), 1)
            prior = dist.MixtureSameFamily(mixture, component)
        else:
            mu, logvar = self.encode(x)
            prior = dist.Normal(mu, logvar)

        structural_embed = dist.ConditionalTransformedDistribution(prior, self.flow_structural)
        energetic_embed = dist.ConditionalTransformedDistribution(structural_embed, self.flow_energetic)
        with pyro.plate("xrd", x.shape[0]):
            z_structural = structural_embed.condition(context_structure).sample()
            z_energetic = energetic_embed.condition(context_energy).sample()

        return self.decode(z_energetic), mu, logvar, z_structural, z_energetic