from codon.base import *
from manifold.model import RiemannianManifoldLinear
import torch.nn.functional as F


class LinearNetwork(BasicModel):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    @property
    def manifold_loss(self) -> torch.Tensor:
        return torch.Tensor(0.0)


class ManifoldLinear(BasicModel):
    def __init__(self):
        super().__init__()
        self.fc1 = RiemannianManifoldLinear(28 * 28, 128)
        self.fc2 = RiemannianManifoldLinear(128, 64)
        self.fc3 = RiemannianManifoldLinear(64, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    @property
    def manifold_loss(self) -> torch.Tensor:
        total_loss = 0.0
        total_loss += self.fc1.compute_loss().factor_loss()
        total_loss += self.fc2.compute_loss().factor_loss()
        total_loss += self.fc3.compute_loss().factor_loss(factor_lap=0.0)

        return total_loss / 3

class ManifoldResidualLinear(BasicModel):
    def __init__(self):
        super().__init__()
        self.fc1 = RiemannianManifoldLinear(28 * 28, 128)
        self.fc2 = RiemannianManifoldLinear(128, 64)
        self.fc3 = RiemannianManifoldLinear(64, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    @property
    def manifold_loss(self) -> torch.Tensor:
        total_loss = 0.0
        total_loss += self.fc1.compute_loss().factor_loss()
        total_loss += self.fc2.compute_loss().factor_loss()
        total_loss += self.fc3.compute_loss().factor_loss(factor_lap=0.0)

        return total_loss / 3
