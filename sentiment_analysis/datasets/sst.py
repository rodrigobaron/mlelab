import pandas as pd
import torch
from torch.utils.data import Dataset


class SSTDataset(Dataset):
    """
    Stanford Sentiment Treebank V1.0
    Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank
    Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts
    Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)
    """

    def __init__(self, file_path, maxlen, tokenizer):
        self.df = pd.read_csv(file_path, delimiter="\t")
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sentence = self.df.loc[index, "sentence"]
        label = self.df.loc[index, "label"]
        tokens = self.tokenizer.tokenize(sentence)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        if len(tokens) < self.maxlen:
            tokens = tokens + ["[PAD]" for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[: self.maxlen - 1] + ["[SEP]"]
        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
        attention_mask = (input_ids != 0).long()
        return input_ids, attention_mask, label
