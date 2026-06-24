from config import *

class WPINN(nn.Module):
    """Coefficient-emitting network (repo-canonical): outputs one wavelet-coefficient vector per
    subdomain (c^-, c^+) plus a trainable bias each. Solution per region: u = W c + b.

    Note: for these *linear* interface problems the W-PINN objective is a convex quadratic in the
    coefficients, so the recommended optimiser is the AD-free least-squares solve used in the
    notebook (sub-second, machine-accurate). This network/Adam path is provided for parity with the
    repo's other examples and for nonlinear extensions."""

    def __init__(self, input_size, num_hidden_layers, hidden_neurons, family_size):
        super(WPINN, self).__init__()
        self.activation = nn.Tanh()

        def make_head():
            layers = [nn.Linear(input_size, hidden_neurons), self.activation]
            for _ in range(num_hidden_layers - 1):
                layers += [nn.Linear(hidden_neurons, hidden_neurons), self.activation]
            layers += [nn.Linear(hidden_neurons, family_size)]
            net = nn.Sequential(*layers)
            for m in net:
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight); init.constant_(m.bias, 0)
            return net

        self.net_minus = make_head()   # -> c^-  (inside)
        self.net_plus  = make_head()   # -> c^+  (outside)
        self.bias_minus = nn.Parameter(torch.tensor(0.0))
        self.bias_plus  = nn.Parameter(torch.tensor(0.0))

    def forward(self, feat):
        f = feat.reshape(-1)
        c_minus = self.net_minus(f)
        c_plus  = self.net_plus(f)
        return c_minus, c_plus, self.bias_minus, self.bias_plus
