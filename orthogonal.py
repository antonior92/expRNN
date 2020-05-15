import torch
import torch.nn as nn

from .parametrization import Parametrization
from .trivializations import cayley_map, expm_skew
from .initialization import henaff_init_, cayley_init_


class Orthogonal(Parametrization):
    """ Class that implements optimization restricted to the Stiefel manifold """
    def __init__(self, input_size, output_size, initializer, mode, param):
        """
        mode: "static" or a tuple such that:
                mode[0] == "dynamic"
                mode[1]: int, K, the number of steps after which we should change the basis of the dyn triv
                mode[2]: int, M, the number of changes of basis after which we should project back onto the manifold the basis. This is particularly helpful for small values of K.
        param: A parametrization of in terms of skew-symmetyric matrices
        """
        super(Orthogonal, self).__init__(input_size, output_size, initializer, mode)
        self.param = param

    def retraction(self, A_raw, base):
        # This could be any parametrization of a tangent space
        A = A_raw.triu(diagonal=1)
        A = A - A.t()
        B = base.mm(self.param(A))
        if self.input_size != self.output_size:
            B = B[:self.input_size, :self.output_size]
        return B

    def project(self, base):
        try:
            # Compute the projection using the thin SVD decomposition
            U, _, V = torch.svd(base, some=True)
            return U.mm(V.t())
        except RuntimeError:
            # If the svd does not converge, fallback to the (thin) QR decomposition
            return torch.qr(base, some=True)[0]


class modrelu(nn.Module):
    def __init__(self, features):
        # For now we just support square layers
        super(modrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)

        return phase * magnitude


class OrthogonalRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, skew_initializer, mode, param):
        super(OrthogonalRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_kernel = Orthogonal(hidden_size, hidden_size, skew_initializer, mode, param=param)
        self.input_kernel = nn.Linear(in_features=self.input_size, out_features=self.hidden_size, bias=False)
        self.nonlinearity = modrelu(hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.input_kernel.weight.data, nonlinearity="relu")

    def default_hidden(self, input):
        return input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

    def forward(self, input, hidden):
        input = self.input_kernel(input)
        hidden = self.recurrent_kernel(hidden)
        out = input + hidden
        out = self.nonlinearity(out)

        return out, out


class OrthogonalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, K, init, rnn_type):
        super(OrthogonalRNN, self).__init__()
        if K != "infty":
            K = int(K)
        # Define init
        if init == 'henaff':
            init_ = henaff_init_
        elif init == 'cayley':
            init_ = cayley_init_
        else:
            raise ValueError("init should be in {'henaff', 'cayley'}")
        # Define mode and param
        if rnn_type == "exprnn":
            mode = "static"
            param = expm_skew
        elif rnn_type == "dtriv":
            # We use 100 as the default to project back to the manifold.
            # This parameter does not really affect the convergence of the algorithms, even for K=1
            mode = ("dynamic", K, 100)
            param = expm_skew
        elif rnn_type == "cayley":
            mode = "static"
            param = cayley_map
        else:
            raise ValueError("Unknown rnn_type")
        self.rnn_cell = OrthogonalRNNCell(input_size, hidden_size, skew_initializer=init_, mode=mode, param=param)

    def forward(self, inputs, state):
        outputs = []
        for input in torch.unbind(inputs, dim=0):
            out_rnn, state = self.rnn_cell(input, state)
            outputs.append(out_rnn)
        return torch.stack(outputs, dim=0), state