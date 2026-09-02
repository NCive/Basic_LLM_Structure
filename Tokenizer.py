from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as blpt
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as bldec

#### Tokeniser Initialise & Training

# tokenizer = Tokenizer(BPE())
# tokenizer.pre_tokenizer = blpt()
# tokenizer.decoder = bldec()

# trainer = BpeTrainer(
#     vocab_size = 8000,
#     min_frequency = 2,
#     show_progress = True,
#     special_tokens = ["<PAD>", "<BOS>", "<EOS>"],
#     initial_alphabet = blpt.alphabet(),
# )

# files = [
#     r"D:\Python Projects\LLM\Tokenizer_TD_Chatgpt.txt",
#     r"D:\Python Projects\LLM\Tokenizer_TD_Claude.txt",
#     r"D:\Python Projects\LLM\Tokenizer_TD_DeepSeek.txt",
#     r"d:\Python Projects\LLM\Tokenizer_TD_Grok.txt"
# ]

# tokenizer.train(files, trainer)
# tokenizer.save("tokenizer.json")
# print("Tokenizer saved.")



###Tokenizer Testing

tokenizer = Tokenizer.from_file("tokenizer.json")
print("Tokenizer loaded.")

# print("Vocabulary size:", tokenizer.get_vocab_size())
# vocab = tokenizer.get_vocab()

# for token, token_id in list(vocab.items())[:20]:
#     print(token_id, token)

text = "Hello, how are you?"
encoded = tokenizer.encode(text)
print("Encoded_id:", encoded.ids)
print("Tokens:", encoded.tokens)

decoded = tokenizer.decode(encoded.ids)
print("Decoded:", decoded)

