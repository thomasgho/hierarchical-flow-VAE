from torch import nn

class GatedDense(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None):
        super(GatedDense, self).__init__()
        self.activation = activation
        self.layer_1 = nn.Linear(in_dim, out_dim)
        self.layer_2 = nn.Linear(in_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.layer_1(x)
        if self.activation is not None:
            h = self.activation(h)
        g = self.sigmoid(self.layer_2(x))
        return h * g

