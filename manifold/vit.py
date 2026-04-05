from manifold.model import RiemannianManifoldLinear, MainfoldLoss
from codon.ops.attention import apply_attention, AttentionOutput

from codon.base import *
from codon.block.mlp import MLP


class ViTBlock(BasicModel):
    def __init__(
        self,
        model_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        linear_type: str = 'linear',
    ):
        super().__init__()

        self.num_heads = num_heads
        if model_dim % num_heads != 0:
            raise ValueError(f'model_dim ({model_dim}) must be divisible by num_heads ({num_heads})')

        self.linear_type = linear_type.lower().strip()
        if not self.linear_type in ['linear', 'manifold']:
            raise ValueError()
        
        # Attention projections (Q, K, V, O)
        self.q_proj: BasicModel = nn.Linear(model_dim, model_dim)
        self.k_proj: BasicModel = nn.Linear(model_dim, model_dim)
        self.v_proj: BasicModel = nn.Linear(model_dim, model_dim)
        self.o_proj: BasicModel = nn.Linear(model_dim, model_dim)

        self.norm_1 = nn.RMSNorm(model_dim)
        self.norm_2 = nn.RMSNorm(model_dim)

        mlp_hidden_dim = int(model_dim * mlp_ratio)
        self.mlp = MLP(
            model_dim,
            mlp_hidden_dim,
            model_dim
        )

        if self.linear_type == 'manifold': self.to_manifold()
    
    def to_manifold(self):
        in_features = self.mlp.in_features
        hidden_features = self.mlp.hidden_features
        out_features = self.mlp.out_features

        self.mlp.fc1 = RiemannianManifoldLinear(in_features, hidden_features)
        self.mlp.fc2 = RiemannianManifoldLinear(hidden_features, out_features)
    
    def forward(self, x: torch.Tensor):
        B, S, C = x.shape
        x_norm = self.norm_1(x)

        head_dim = C // self.num_heads

        q = self.q_proj(x_norm).view(B, S, self.num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, S, self.num_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, S, self.num_heads, head_dim).transpose(1, 2)

        output: AttentionOutput = apply_attention(
            query_states=q,
            key_states=k,
            value_states=v,
            is_causal=False
        )
        
        attn_out = output.output.transpose(1, 2).contiguous().view(B, S, C)
        x = x + self.o_proj(attn_out)

        x_norm = self.norm_2(x)
        x = x + self.mlp(x_norm)

        return x


class ViT(BasicModel):
    '''
    Vision Transformer (ViT) architecture.

    Attributes:
        patch_embed (nn.Conv2d): The patch embedding layer.
        cls_token (nn.Parameter): The classification token.
        pos_embed (nn.Parameter): The positional embeddings.
        blocks (nn.ModuleList): The sequence of Transformer blocks.
        norm (nn.RMSNorm): The final normalization layer.
        head (nn.Linear): The classification head.
    '''

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        model_dim: int = 768,
        num_heads: int = 12,
        depth: int = 12,
        mlp_ratio: int = 4,
        linear_type: str = 'linear',
    ) -> None:
        '''
        Initializes the ViT model.

        Args:
            img_size (int): Size of the input image.
            patch_size (int): Size of each patch.
            in_channels (int): Number of input channels.
            num_classes (int): Number of classes for classification.
            model_dim (int): Dimensionality of the embeddings.
            num_heads (int): Number of attention heads.
            depth (int): Number of Transformer blocks.
            mlp_ratio (int): Expansion ratio for the MLP hidden dimension.
            linear_type (str): Type of linear layer ('linear' or 'manifold').
        '''
        super().__init__()
        
        if img_size % patch_size != 0:
            raise ValueError(f'img_size ({img_size}) must be divisible by patch_size ({patch_size})')
            
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_channels, model_dim, kernel_size=patch_size, stride=patch_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, model_dim))
        
        self.blocks = nn.ModuleList([
            ViTBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                linear_type=linear_type
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.RMSNorm(model_dim)
        self.head = nn.Linear(model_dim, num_classes)
        
        self._init_weights()
        
    def _init_weights(self) -> None:
        '''
        Initializes the weights for the pos_embed and cls_token.
        '''
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Defines the computation performed at every call.

        Args:
            input_tensor (torch.Tensor): Input images of shape (batch_size, in_channels, img_size, img_size).

        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes).
        '''
        batch_size = input_tensor.shape[0]
        
        # Patch embedding: [B, C, H, W] -> [B, D, H/P, W/P] -> [B, D, N] -> [B, N, D]
        x = self.patch_embed(input_tensor)
        x = x.flatten(2).transpose(1, 2)
        
        # Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Classification head
        cls_output = x[:, 0]
        cls_output = self.norm(cls_output)
        output = self.head(cls_output)
        
        return output
    
    @property
    def manifold_loss(self) -> torch.Tensor:
        '''
        Computes the accumulated manifold loss (cosine + Laplacian) for all manifold layers.

        Returns:
            torch.Tensor: The total calculated manifold loss. Returns 0.0 if no manifold layers exist.
        '''
        total_cos = 0.0
        total_lap = 0.0
        has_manifold = False

        for module in self.modules():
            if hasattr(module, 'compute_loss') and callable(module.compute_loss):
                loss: MainfoldLoss = module.compute_loss()
                if not has_manifold:
                    total_cos = loss.cosine
                    total_lap = loss.laplacian
                    has_manifold = True
                else:
                    total_cos = total_cos + loss.cosine
                    total_lap = total_lap + loss.laplacian
        
        if not has_manifold:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
            
        return MainfoldLoss(cosine=total_cos, laplacian=total_lap).factor_loss()


def vit_cifar100(linear_type: str = 'linear') -> ViT:
    '''
    Constructs a shallower ViT model specifically tailored for CIFAR-100 (32x32 images).

    Args:
        linear_type (str): Type of linear layer to use ('linear' or 'manifold').

    Returns:
        ViT: The initialized shallower ViT model for CIFAR-100.
    '''
    return ViT(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=100,
        model_dim=384,
        num_heads=12,
        depth=6,
        mlp_ratio=4,
        linear_type=linear_type
    )
