from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import multiprocessing
import os


"""
Create a dataset class for the plot generator
"""
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PlotGeneratorDataset(Dataset):
    def __init__(
        self,
        df=pd.read_csv("./data/processed.csv", sep=","),
        tokenizer=AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neo-2.7B",
            pad_token="[PAD]",
        ),
    ):
        self.tokenizer = tokenizer
        # self.max_length = max([len(tokenizer.encode(txt)) for txt in df["text"]])
        self.df = df
        self.input_ids = []
        self.attention_mask = []
        self.labels = []
        for txt in df["text"]:
            encoded = self.tokenizer(
                txt,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_attention_mask=True,
                return_tensors="pt",
            )
            self.input_ids.append(encoded["input_ids"])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index], self.labels[index]


def split_dataset(dataset):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    return train_ds, val_ds


if __name__ == "__main__":
    dataset = PlotGeneratorDataset()
    train_ds, val_ds = train_test_split(dataset, test_size=0.2)
    print(len(train_ds))
    print(len(val_ds))
