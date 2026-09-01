import torch
import torch.nn as nn
from Embedding import final_embedded
from Attention_Block import TransformerBlock


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



model = Transformer(embedding_dim = 256, num_heads = 8, num_layers = 4)

output = model(final_embedded)
print("Input shape:", final_embedded.shape)
print("Output shape:", output.shape)


