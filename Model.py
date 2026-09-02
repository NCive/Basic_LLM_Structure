import torch
import torch.nn as nn
from Transformer import Transformer


class GPTModel(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embedding_dim,
                 max_seq_length, 
                 num_layers, 
                 num_heads):

        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.transformer = Transformer(embedding_dim, num_heads, num_layers)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)


    def forward(self, input_ids):

        batch_size, seq_length = input_ids.shape

        token_embeddings = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_length, 
                                    device=input_ids.device).unsqueeze(0)
        position_embeddings = self.position_embedding(position_ids)
        x = token_embeddings + position_embeddings

        x = self.transformer(x)
        logits = self.lm_head(x)

        return logits


