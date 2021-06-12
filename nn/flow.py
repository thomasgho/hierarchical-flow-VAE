from torchdyn.models import *


class Flow(nn.Module):
    def __init__(self, dim):
        super(Flow, self).__init__()
        self.vec_field = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.Softplus(),
            nn.Linear(dim*2, dim*4),
            nn.Softplus(),
            nn.Linear(dim*4, dim*2),
            nn.Softplus(),
            nn.Linear(dim*2, dim))

    def trace(self, x_out, x_in):
        trJ = 0.
        for i in range(x_in.shape[1]):
            trJ += torch.autograd.grad(x_out[:, i].sum(), x_in, allow_unused=False, create_graph=True)[0][:, i]
        return trJ

    def forward(self, x):
        with torch.set_grad_enabled(True):
            x_in = torch.autograd.Variable(x[: ,1:], requires_grad=True).to(x)
            x_out = self.vec_field(x_in)
            trJ = self.trace(x_out, x_in)
        return torch.cat([-trJ[:, None], x_out], 1) + 0*x  # `+ 0*x` has the only purpose of connecting x[:, 0] to autograd graph
