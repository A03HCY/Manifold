from codon.base import *
from manifold.model import RiemannianManifoldConv2d


class ConvNetwork(BasicModel):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Block 1: 3x32x32 -> 32x16x16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: 32x16x16 -> 64x8x8
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 64x8x8 -> 128x4x4
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 128x4x4 -> 256x2x2
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 分类头
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.act5 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))
        x = self.pool4(self.act4(self.conv4(x)))
        
        x = self.flatten(x)
        x = self.act5(self.fc1(x))
        out = self.fc2(x)
        return out
    
    @property
    def manifold_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)


class RiemannianConvNetwork(BasicModel):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Block 1
        self.conv1 = RiemannianManifoldConv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1, 
            kappa_init=2.0, lambda_init=0.1, scale_init=10.0
        )
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2 = RiemannianManifoldConv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1,
            kappa_init=2.0, lambda_init=0.1, scale_init=10.0
        )
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv3 = RiemannianManifoldConv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1,
            kappa_init=2.0, lambda_init=0.1, scale_init=10.0
        )
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4
        self.conv4 = RiemannianManifoldConv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1,
            kappa_init=2.0, lambda_init=0.1, scale_init=10.0
        )
        self.act4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.act5 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))
        x = self.pool4(self.act4(self.conv4(x)))
        
        x = self.flatten(x)
        x = self.act5(self.fc1(x))
        out = self.fc2(x)
        return out
    
    @property
    def manifold_loss(self) -> torch.Tensor:
        total_loss = 0.0
        
        total_loss += self.conv1.compute_loss().factor_loss()
        total_loss += self.conv2.compute_loss().factor_loss()
        total_loss += self.conv3.compute_loss().factor_loss()
        total_loss += self.conv4.compute_loss().factor_loss()
        
        return total_loss / 4