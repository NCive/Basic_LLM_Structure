import torch
import torch.nn as nn

### First Transformer + Multi-head Attention + LayerNorm 
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        #Layernorm1
        self.layer_norm1 = nn.LayerNorm(embedding_dim)

        #QKV
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        #Layernorm2
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        #FFN
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim*4),
            nn.GELU(),
            nn.Linear(embedding_dim*4, embedding_dim)
        )

    ###Forwrad Pass
    def forward(self,x):

        batch_size, seq_length, _ = x.shape

        normalized_1 = self.layer_norm1(x)

        Q = self.query(normalized_1)
        K = self.key(normalized_1)
        V = self.value(normalized_1)


        #QKV_Split
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim)


        #For Attention Scores on each head independently
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)


        #Attention Score
        scores = torch.matmul(Q,K.transpose(-1,-2))
        scores = scores / self.head_dim**0.5 #normalise


        #causal mask
        causal_mask = torch.tril(
            torch.ones(
                seq_length,
                seq_length,
                device = x.device
            )
        )


        #So that softmax scores 0 on these
        scores = scores.masked_fill(
            causal_mask == 0,
            float("-inf")
        )


        #Attention weight
        attention_weights = torch.softmax(scores, dim=-1)


        #Final attention output
        attention_output = torch.matmul(attention_weights, V)


        #Concatenate
        attention_output = attention_output.transpose(1,2)
        attention_output = attention_output.contiguous().view(
            batch_size,
            seq_length,
            self.embedding_dim
        )


        #Residual1
        residual1 = x + attention_output
        normalized_2 = self.layer_norm2(residual1)


        #FFN
        ffn_output = self.ffn(normalized_2)


        #Residual2
        final_output = ffn_output + residual1


        return final_output
