import torch
from tokenizers import Tokenizer
from Model import GPTModel

vocab_size = 3147 
embedding_dim = 256 
max_seq_length = 128 
num_layers = 4 
num_heads = 8


tokenizer = Tokenizer.from_file("tokenizer.json")
model = GPTModel( vocab_size=vocab_size, 
                 embedding_dim=embedding_dim, 
                 max_seq_length=max_seq_length, 
                 num_layers=num_layers, 
                 num_heads=num_heads)
model.load_state_dict(torch.load("model.pt"))
model.eval()

prompt = "The Future of AI"
print(f"User: {prompt}")
encoded = tokenizer.encode(prompt)
input_ids = torch.tensor(encoded.ids, dtype=torch.long).unsqueeze(0)

max_tokens = 50

with torch.no_grad():
    for _ in range(max_tokens):
        input_context = input_ids[:, -max_seq_length:]
        logits = model(input_context)
        next_token_logits = logits[:, -1, :]

        temperature = 0.8
        next_token_logits = next_token_logits / temperature
        probabilities = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        
        input_ids = torch.cat([input_ids, next_token], dim=1)

response = tokenizer.decode(input_ids[0].tolist())

print(f"Response: {response}")