import torch 
import torch.nn as nn
from Dataset import LanguageModelDataset
from torch.utils.data import DataLoader
from Model import GPTModel


vocab_size = 3147
embedding_dim = 256
max_seq_length = 128
num_layers = 4
num_heads = 8

token_ids = torch.load("training_tokens.pt")
dataset = LanguageModelDataset(token_ids, max_seq_length)

dataloader = DataLoader(dataset, batch_size = 4, shuffle = True)


model = GPTModel(vocab_size = vocab_size,
                 embedding_dim = embedding_dim, 
                 max_seq_length = max_seq_length, 
                 num_layers = num_layers, 
                 num_heads = num_heads)
loss_ffn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)

max_steps = 500
model.train()

for step, (input_batch, target_batch) in enumerate(dataloader):
    optimizer.zero_grad()
    logits = model(input_batch)

    loss = loss_ffn(
    logits.reshape(-1, vocab_size),
    target_batch.reshape(-1)
    )

    loss.backward()
    optimizer.step()

    if step % 50 ==0:
        print(f"Loss for Step {step}: {loss.item():.4f}")
    if step >= max_steps:
        break

torch.save(model.state_dict(), "model.pt")

print("Model saved to model.pt")






