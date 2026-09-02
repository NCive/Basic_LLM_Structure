# import torch
# import torch.nn as nn
# from Embedding import final_embedded
# from Transformer import Transformer
# from tokenizers import Tokenizer


# embedding_dim = 256
# vocab_size = 3147

# tokenizer = Tokenizer.from_file("tokenizer.json")
# model = Transformer(embedding_dim, num_heads = 8, num_layers = 4)

# transformer_output = model(final_embedded)
# print("Transformer output:", transformer_output.shape)


# lm_head = nn.Linear(embedding_dim, vocab_size)
# logits = lm_head(transformer_output)
# print("Logits shape:", logits.shape)


# last_token_logits = logits[:, -1, :]
# print("Last token logits shape:", last_token_logits.shape)


# probabilities = torch.softmax(last_token_logits, dim=-1)
# print("Probabilities shape:", probabilities.shape)


# predicted_id = torch.argmax(probabilities, dim=-1)
# print("Predicted token ID:", predicted_id)


# predicted_token = tokenizer.decode(predicted_id.tolist())
# print("Predicted token:", predicted_token)