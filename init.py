from config import *

from manifold.linear import LinearNetwork, ManifoldLinear
from manifold.conv import ConvNetwork, RiemannianConvNetwork

from torch.optim import Adam
from torch import nn

baseline_linear_model = LinearNetwork().to(device)
manifold_linear_model = ManifoldLinear().to(device)

baseline_linear_opt = Adam(baseline_linear_model.parameters(), lr=leaning_rate)
manifold_linear_opt = Adam(manifold_linear_model.parameters(), lr=leaning_rate)

baseline_linear_cri = nn.CrossEntropyLoss()
manifold_linear_cri = nn.CrossEntropyLoss()

baseline_conv_model = ConvNetwork().to(device)
manifold_conv_model = RiemannianConvNetwork().to(device)

baseline_conv_opt = Adam(baseline_conv_model.parameters(), lr=leaning_rate)
manifold_conv_opt = Adam(manifold_conv_model.parameters(), lr=leaning_rate)

baseline_conv_cri = nn.CrossEntropyLoss()
manifold_conv_cri = nn.CrossEntropyLoss()
