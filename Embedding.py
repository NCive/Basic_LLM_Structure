# import torch 
# import torch.nn as nn
# from tokenizers import Tokenizer

# vocab_size = 3147 
# embedding_dim = 256
# max_seq_length = 512

# ### Tokens to Embedding 

# embedding_layer = nn.Embedding(vocab_size, embedding_dim)
# positional_embedding = nn.Embedding(max_seq_length, embedding_dim)


# tokenizer = Tokenizer.from_file("tokenizer.json")


# text = "Hello, how are you?"
# encoded = tokenizer.encode(text)
# input_ids = torch.tensor(encoded.ids).unsqueeze(0)
# # print("Input IDs:", input_ids)
# # print("Input shape:", input_ids.shape)


# embedded = embedding_layer(input_ids)
# # print("Embedded shape:", embedded.shape)


# seq_length = input_ids.shape[1]
# position_ids = torch.arange(seq_length).unsqueeze(0)
# # print("Position IDs:", position_ids)
# position_embed = positional_embedding(position_ids)
# # print("Positional embedding shape:", position_embed.shape)


# final_embedded = embedded + position_embed
# # print("Final embedding:", final_embedded)
# # print("Final embedding shape:", final_embedded.shape)
