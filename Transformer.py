# import torch
import torch.nn as nn
from Attention_Block import TransformerBlock

### N * Transformer Layers

class Transformer(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim = embedding_dim,
                num_heads = num_heads
            )
            for _ in range(num_layers)
        ])

        self.final_layer_norm = nn.LayerNorm(embedding_dim)


    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_layer_norm(x)

        return x


