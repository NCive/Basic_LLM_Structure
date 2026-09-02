# Building a Small Language Model From Scratch

A beginner-friendly walkthrough of building, training, and running a tiny GPT-style language model from scratch in PyTorch — no pretrained weights, no shortcuts. This README documents the full pipeline and the experiment used to validate it.

> **Goal of this experiment:** not to build a *good* model, but to build a *correct, working pipeline* — tokenizer → embeddings → Transformer → training → text generation — on a small corpus, so every piece can be verified before scaling up.

---

## Overview: The Full Pipeline

```
Raw Text
   ↓
Tokenizer (Byte-Level BPE)
   ↓
Token IDs
   ↓
Token Embedding + Positional Embedding
   ↓
Transformer Block × N
   (LayerNorm → Multi-Head Causal Self-Attention → Residual
    → LayerNorm → Feed-Forward Network → Residual)
   ↓
Final LayerNorm
   ↓
LM Head (Linear layer → vocabulary logits)
   ↓
Next-Token Prediction
```

This is a **decoder-only Transformer** — the same family of architecture behind GPT-style models, just at a scale small enough to train and run on a laptop CPU.

---

## 1. The Tokenizer

Before any model training happens, we need a **tokenizer**: something that converts raw text into numbers (token IDs) the model can process.

### What a tokenizer actually is

- **Vocabulary** — a list of all recognized tokens and their ID numbers
- **Merge rules** — rules for how sub-word pieces combine (Byte-Pair Encoding, or BPE)
- Saved as `tokenizer.json` (plus config files)

### Training order matters

The tokenizer is trained **before** the LLM, on raw text, and then **frozen**:

```
Raw corpus → train BPE tokenizer → vocabulary + merge rules → save (frozen)
                                              ↓
                                    tokenize LLM training data → train LLM
```

⚠️ **Critical rule:** the same tokenizer must be used for training and inference. Retraining the tokenizer reshuffles every token ID and breaks any model already trained against the old vocabulary.

### Tokenizer dataset vs. LLM dataset

These are easy to confuse:
- **Tokenizer dataset** → "what tokens should exist?" (raw text → vocab/merges)
- **LLM dataset** → "what should the model learn?" (tokenized text → next-token prediction → weights)

You can use the same source text for both, but the processing and purpose differ.

### What makes a good tokenizer corpus

- **Match your real inference distribution** — if your model will see code, structured prompts, or multiple languages, the tokenizer corpus needs those shapes too, or that content gets fragmented into inefficient token sequences later.
- **Diverse, not just large** — merges are frequency-based, so underrepresented content (a language, a domain) ends up split into many small, awkward tokens.
- **Deduplicated** — repeated documents distort merge statistics.
- **Preserve punctuation, numbers, and whitespace** — don't over-clean; real text has typos, mixed case, and meaningful formatting.
- **Size is modest** — tokenizer training is much cheaper than model training. 100 MB–500 MB is plenty for a first experiment.

### Implementation (Byte-Level BPE)

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

bpe = BPE()
tokenizer = Tokenizer(bpe)
tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(
    vocab_size=16000,
    min_frequency=2,
    show_progress=True,
    special_tokens=["<PAD>", "<BOS>", "<EOS>"],
)

tokenizer.train(files=["corpus.txt"], trainer=trainer)
tokenizer.save("tokenizer.json")
```

Key parameter notes:
- `vocab_size` is the **target total** vocabulary size (byte alphabet + special tokens + learned merges combined), not "16k merges on top of everything else."
- No `<UNK>` token is needed — byte-level BPE can represent *any* input as raw bytes, so nothing is ever truly "unknown."
- Vocabulary size has a direct, concrete cost: `vocab_size × embedding_dim` = the size of the token embedding table. E.g. `16,000 × 256 = 4,096,000` parameters just for embeddings.

---

## 2. Turning Text Into Model Input (Embeddings)

```python
embedding_layer = nn.Embedding(vocab_size, embedding_dim)         # token ID → vector
positional_embedding = nn.Embedding(max_seq_length, embedding_dim) # position → vector

encoded = tokenizer.encode(text)
input_ids = torch.tensor(encoded.ids).unsqueeze(0)   # add batch dimension

embedded = embedding_layer(input_ids)                 # [batch, tokens] → [batch, tokens, dim]
position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
position_embed = positional_embedding(position_ids)

final_embedded = embedded + position_embed             # token meaning + position info
```

**Why add position embeddings?** Self-attention has no built-in sense of word order — without positional information, "the cat sat" and "sat cat the" would look identical to the model. Adding a learned position vector to each token embedding gives the model that missing sense of order.

```
"Hello, how are you?"
        ↓ Tokenizer
     [1, 9] token IDs
        ↓ Token Embedding
     [1, 9, 256]
        + Position Embedding
     [1, 9, 256]
        ↓
final_embedded → ready for Self-Attention
```

---

## 3. Self-Attention: The Core Mechanism

### Query, Key, Value

Self-attention creates three learned projections of the same input:

```python
Q = query_layer(final_embedded)   # "what am I looking for?"
K = key_layer(final_embedded)     # "what do I contain?"
V = value_layer(final_embedded)   # "what should I contribute?"
```

All three come from the **same** input — that's what makes it *self*-attention. Each token's Query is compared against every other token's Key to decide how much to attend to it, and that determines how much of each token's Value gets pulled into the output.

### The attention calculation

```python
scores = torch.matmul(Q, K.transpose(-2, -1))   # every token vs. every token
scores = scores / (head_dim ** 0.5)               # scale for numerical stability

attention_weights = torch.softmax(scores, dim=-1)  # turn scores into probabilities (sum to 1)
attention_output = torch.matmul(attention_weights, V)  # weighted mix of Values
```

### Causal masking

For a *decoder*, a token must never see future tokens — only what came before it. This is enforced with a lower-triangular mask:

```
        Key
        0   1   2   3
Query
0       ✓   X   X   X
1       ✓   ✓   X   X
2       ✓   ✓   ✓   X
3       ✓   ✓   ✓   ✓
```

```python
causal_mask = torch.tril(torch.ones(seq_length, seq_length))
scores = scores.masked_fill(causal_mask == 0, float("-inf"))
```

Setting disallowed positions to `-inf` makes softmax turn them into ~0 probability, so a token can only draw information from itself and earlier tokens.

### Multi-Head Attention

Instead of one large attention calculation, the embedding dimension is split into multiple smaller **heads** that run attention in parallel, then get recombined:

```
Input               [1, 9, 256]
  ↓ Q/K/V projections
                     [1, 9, 256]
  ↓ split into 8 heads
                     [1, 9, 8, 32]
  ↓ move heads next to batch dim
                     [1, 8, 9, 32]
  ↓ attention per head (Q×Kᵀ, scale, mask, softmax, ×V)
                     [1, 8, 9, 32]
  ↓ recombine heads
                     [1, 9, 256]
```

Each of the 8 heads works on a smaller 32-dimensional slice, letting the model attend to different kinds of relationships (e.g. grammar vs. meaning) simultaneously.

---

## 4. The Transformer Block

One block combines attention with a feed-forward network, using **pre-norm** structure (normalize before each sub-layer) and residual (skip) connections:

```
Input [1, 9, 256]
  ↓ LayerNorm
  ↓ Multi-Head Causal Self-Attention
  ↓ Residual Add (+ original input)
  ↓ LayerNorm
  ↓ Feed-Forward Network (256 → 1024 → 256, with GELU)
  ↓ Residual Add
  ↓ Output [1, 9, 256]
```

- **LayerNorm** stabilizes each token's values before each sub-layer — shape is unchanged, but it has learnable scale/shift parameters.
- **Residual connections** (`x + sublayer(x)`) keep gradients flowing cleanly through deep networks and preserve the original signal alongside what each layer learns.
- **The FFN** expands each token's representation (256 → 1024), applies a non-linearity (GELU), then projects back down (1024 → 256) — this is where most of the model's per-token "reasoning capacity" lives.

Multiple blocks are stacked to build depth:

```python
class Transformer(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads) for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final_layer_norm(x)
```

---

## 5. The LM Head

The final step converts each token's 256-dimensional representation into a score for **every possible vocabulary token**:

```python
lm_head = nn.Linear(embedding_dim, vocab_size)
logits = lm_head(transformer_output)   # [1, 9, 256] → [1, 9, 3147]
```

`logits[:, -1, :]` — the scores at the **last** position — are what matter for predicting the next token. Softmax turns these into probabilities, and `argmax` (or sampling) picks the actual next token.

> At this stage, predictions are meaningless — every weight in the model is still randomly initialized. The architecture is complete; nothing has *learned* anything yet.

---

## 6. Assembling the Full Model

All the individual pieces (embeddings, Transformer blocks, LM head) get combined into a single reusable `nn.Module` that accepts any batch of token IDs — not just one fixed test sentence:

```python
class GPTModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_length, num_layers, num_heads):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.transformer = Transformer(embedding_dim, num_heads, num_layers)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids):
        batch_size, seq_length = input_ids.shape
        token_embeddings = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.position_embedding(position_ids)

        x = token_embeddings + position_embeddings
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits
```

```
input_ids [batch, seq_len]
   ↓ Token + Position Embedding
[batch, seq_len, 256]
   ↓ N × Transformer Block (includes final LayerNorm)
[batch, seq_len, 256]
   ↓ LM Head
[batch, seq_len, vocab_size]
```

---

## 7. Preparing Training Data

### The experiment corpus

For this experiment, four small text files (~1 MB total) were used as the training corpus — enough to validate the pipeline end-to-end, though far too small to produce a genuinely capable model.

### Tokenize once, save to disk

```python
tokenizer = Tokenizer.from_file("tokenizer.json")
corpus = "".join(open(f, encoding="utf-8").read() + "\n" for f in files)

token_ids = torch.tensor(tokenizer.encode(corpus).ids, dtype=torch.long)
torch.save(token_ids, "training_tokens.pt")
```

Tokenizing once and saving the result avoids re-tokenizing the raw text on every training run.

**Result:** 335,515 total tokens.

### Building (input, target) pairs

The token stream is cut into overlapping windows. Each training example is a sequence and that same sequence shifted forward by one token — this is what teaches **next-token prediction**:

```
Input:  t1  t2  t3 ... t128
Target: t2  t3  t4 ... t129
```

```python
class LanguageModelDataset(Dataset):
    def __init__(self, token_ids, seq_length):
        self.token_ids = token_ids
        self.seq_length = seq_length

    def __len__(self):
        return len(self.token_ids) - self.seq_length

    def __getitem__(self, index):
        input_ids = self.token_ids[index : index + self.seq_length]
        target_ids = self.token_ids[index + 1 : index + self.seq_length + 1]
        return input_ids, target_ids

dataset = LanguageModelDataset(token_ids, seq_length=128)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

With 335,515 tokens and a sequence length of 128, this produces **335,387 overlapping samples** — not 335,387 separate documents, but sliding windows (`tokens 0–127 → 1–128`, `tokens 1–128 → 2–129`, etc.). `DataLoader` groups these into batches (`[4, 128]`) and shuffles them each epoch.

---

## 8. Training

One training step follows the standard supervised-learning loop:

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

logits = model(input_batch)                                        # forward pass
loss = loss_fn(logits.reshape(-1, vocab_size), target_batch.reshape(-1))

optimizer.zero_grad()   # clear old gradients
loss.backward()         # compute new gradients
optimizer.step()        # update weights
```

This is wrapped in a loop over the `DataLoader` for a fixed number of steps:

```python
for step, (input_batch, target_batch) in enumerate(dataloader):
    optimizer.zero_grad()
    logits = model(input_batch)
    loss = loss_fn(logits.reshape(-1, vocab_size), target_batch.reshape(-1))
    loss.backward()
    optimizer.step()

    if step >= max_steps:
        break

torch.save(model.state_dict(), "model.pt")
```

### Experiment result

Over 500 training steps, loss dropped from **8.15 → 0.20**, with occasional upward blips (expected — each batch has different difficulty). This confirmed the entire training pipeline was wired correctly: **Dataset → DataLoader → Model → Logits → Loss → Backprop → Optimizer → decreasing loss.**

> ⚠️ Because the training corpus is so small, loss falling this low means the model largely **memorized** the corpus rather than learning general language patterns. That's an expected and acceptable outcome for this experiment — the goal was validating the pipeline, not producing a capable model.

`model.state_dict()` saves only the **learned parameters** (embeddings, attention weights, LayerNorm params, FFN weights, LM head) — not the architecture. Loading it later requires recreating the same `GPTModel` class first.

---

## 9. Text Generation

With a trained (or even partially trained) model, text can be generated autoregressively — one token at a time, feeding each prediction back in as input for the next step:

```python
model.load_state_dict(torch.load("model.pt"))
model.eval()

input_ids = torch.tensor(tokenizer.encode(prompt).ids).unsqueeze(0)

with torch.no_grad():
    for _ in range(max_new_tokens):
        input_context = input_ids[:, -max_seq_length:]   # keep context bounded
        logits = model(input_context)
        next_token_logits = logits[:, -1, :]               # only the last position matters

        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=1)

generated_text = tokenizer.decode(input_ids[0].tolist())
```

### Greedy decoding vs. sampling

- **Argmax (greedy)** always picks the single highest-probability token — simple, deterministic, but prone to repetition loops (`"and, and, and..."`).
- **Temperature sampling** introduces controlled randomness:

  ```python
  next_token_logits = next_token_logits / temperature   # e.g. 0.8
  probabilities = torch.softmax(next_token_logits, dim=-1)
  next_token = torch.multinomial(probabilities, num_samples=1)
  ```

  This reduces repetitive loops, though it can't fix the underlying problem of too little training data.

### Experiment outcome

Generated text was repetitive and often nonsensical — expected given only 500 training steps on ~335k tokens. The important result: the **full pipeline worked end-to-end** — tokenize → train → save/load weights → generate text autoregressively.

---

## Summary: What This Experiment Validated

| Component | Status |
|---|---|
| Byte-level BPE tokenizer (train + save + load) | ✅ Working |
| Token + positional embeddings | ✅ Working |
| Causal multi-head self-attention | ✅ Working |
| Full pre-norm Transformer block (attention + FFN + residuals) | ✅ Working |
| Stacking N Transformer blocks | ✅ Working |
| LM head → vocabulary logits | ✅ Working |
| Training loop (loss computation, backprop, optimizer updates) | ✅ Working — loss 8.15 → 0.20 |
| Save/load model checkpoints | ✅ Working |
| Autoregressive text generation (greedy + sampling) | ✅ Working |

## Known Limitations & Next Steps

- **Training data is far too small** (~335k tokens) for a genuinely useful model — this was intentional, to keep experiments fast while validating the pipeline. Scaling up the corpus (ideally hundreds of MB to GB, diverse in domain and language) is the top priority before expecting real capability.
- **Generation quality** can be improved further with temperature + top-k / top-p sampling on top of what's already implemented.
- **The tokenizer** was trained on the same small corpus — a production tokenizer would use a larger, more representative dataset matching the model's intended use case.
- This is a **single-head-of-understanding build**: no advanced techniques (MoE, hybrid local/global attention, RAG, KV-caching for fast inference) have been added yet. The architecture here is intentionally the minimal, standard decoder-only Transformer, to be extended later once the fundamentals are solid.

---

*Built as a learning project to understand every stage of a language model — from raw text to generated text — by implementing it manually rather than starting from a pretrained checkpoint.*