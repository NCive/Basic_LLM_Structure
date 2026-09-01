import torch
import torch.nn as nn

from Embedding import final_embedded
from Transformer import Transformer

embedding_dim = 256
vocab_size = 3147

model = Transformer(embedding_dim, num_heads = 8, num_layers = 4)

transformer_output = model(final_embedded)
print("Transformer output:", transformer_output.shape)


lm_head = nn.Linear(embedding_dim, vocab_size)
logits = lm_head(transformer_output)
print("Logits shape:", logits.shape)