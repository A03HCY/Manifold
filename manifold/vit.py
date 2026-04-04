from manifold.model import RiemannianManifoldLinear
from codon.ops.attention import apply_attention, AttentionOutput

from codon.base import *
from codon.block.mlp import MLP


class ViTBlock(BasicModel):
    def __init__(
        self,
        model_dim: int = 768,
        mlp_ratio: int = 4,
        linear_type: str = 'linear',
    ):
        super().__init__()

        self.linear_type = linear_type.lower().strip()
        if not self.linear_type in ['linear', 'manifold']:
            raise ValueError()
        
        LinearClass = RiemannianManifoldLinear if self.linear_type == 'manifold' else nn.Linear

        # Attention projections (Q, K, V, O)
        self.q_proj: BasicModel = LinearClass(model_dim, model_dim)
        self.k_proj: BasicModel = LinearClass(model_dim, model_dim)
        self.v_proj: BasicModel = LinearClass(model_dim, model_dim)
        self.o_proj: BasicModel = LinearClass(model_dim, model_dim)

        self.norm_1 = nn.RMSNorm()
        self.norm_2 = nn.RMSNorm()

        mlp_hidden_dim = int(model_dim * mlp_ratio)
        self.mlp = MLP(
            model_dim,
            mlp_hidden_dim,
            model_dim
        )
    
    def forward(self, x: torch.Tensor):
        x_norm = self.norm_1(x)

        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        output: AttentionOutput = apply_attention(
            query_states=q,
            key_states=k,
            value_states=v,
            is_causal=False
        )
        x += self.o_proj(output.output)

        x_norm = self.norm_2(x)
        x += self.mlp(x_norm)

        return x