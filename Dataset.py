# import torch
# from tokenizers import Tokenizer
from torch.utils.data import Dataset

### LLM Training Dataset Creation
##Instead of reading and tokenizing everytime, we create a tokenized file

# tokenizer = Tokenizer.from_file("tokenizer.json")

# files = [
#     r"D:\Python Projects\LLM\Tokenizer_TD_Chatgpt.txt",
#     r"D:\Python Projects\LLM\Tokenizer_TD_Claude.txt",
#     r"D:\Python Projects\LLM\Tokenizer_TD_DeepSeek.txt",
#     r"d:\Python Projects\LLM\Tokenizer_TD_Grok.txt"
# ]

# corpus = ""

# for file in files:
#     with open(file, "r", encoding="utf-8") as f:
#         corpus += f.read()
#         corpus += "/n"

# print("Corpus characters:", len(corpus))


# encoded = tokenizer.encode(corpus)
# token_ids = encoded.ids
# print("Total Tokens:", len(token_ids))


# token_ids = torch.tensor(token_ids, dtype=torch.long)
# print("token Tensor Shape:", token_ids.shape)


# torch.save(token_ids, "training_tokens.pt")
# print("training_tokens.pt saved")



class LanguageModelDataset(Dataset):

    def __init__(self, token_ids, seq_length):
        self.token_ids = token_ids
        self.seq_length = seq_length

    def __len__(self):
        return len(self.token_ids) - self.seq_length

    def __getitem__(self, index):

        input_ids = self.token_ids[
            index : index + self.seq_length
        ]

        target_ids = self.token_ids[
            index + 1 : index + self.seq_length + 1
        ]

        return input_ids, target_ids

