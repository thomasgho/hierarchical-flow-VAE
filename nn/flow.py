from torchdyn.models import *
from torch.distributions import MultivariateNormal


class Flow(nn.Module):
    def __init__(self, dim, trace_estimator):
        super(Flow, self).__init__()
        self.vec_field = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.Softplus(),
            nn.Linear(dim*2, dim*4),
            nn.Softplus(),
            nn.Linear(dim*4, dim*2),
            nn.Softplus(),
            nn.Linear(dim*2, dim))
        if trace_estimator == 'autograd':
            self.trace_estimator = self.autograd_trace
            self.noise_dist, self.noise = None, None
        elif trace_estimator == 'hutchinson':
            self.trace_estimator = self.hutch_trace
            self.noise_dist, self.noise = MultivariateNormal(torch.zeros(dim), torch.eye(dim)), None
        else:
            raise NotImplementedError

    def autograd_trace(self, x_out, x_in, **kwargs):
        trJ = 0.
        for i in range(x_in.shape[1]):
            trJ += torch.autograd.grad(x_out[:, i].sum(), x_in, allow_unused=False, create_graph=True)[0][:, i]
        return trJ

    def hutch_trace(self, x_out, x_in, noise=None):
        jvp = torch.autograd.grad(x_out, x_in, noise, create_graph=True)[0]
        trJ = torch.einsum('bi,bi->b', jvp, noise)
        return trJ

    def forward(self, x):
        with torch.set_grad_enabled(True):
            x_in = torch.autograd.Variable(x[: ,1:], requires_grad=True).to(x)
            x_out = self.vec_field(x_in)
            trJ = self.trace_estimator(x_out, x_in, noise=self.noise)
        return torch.cat([-trJ[:, None], x_out], 1) + 0*x  # `+ 0*x` has the only purpose of connecting x[:, 0] to autograd graph
