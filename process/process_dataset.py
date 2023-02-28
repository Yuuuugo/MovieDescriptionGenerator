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
        tokenizer=AutoTokenizer.from_pretrained("gpt2-medium"),
    ):
        self.tokenizer = tokenizer
        # self.max_length = max([len(tokenizer.encode(txt)) for txt in df["text"]])
        self.df = df
        self.input_ids = []
        self.attention_mask = []
        self.labels = []
        for title in df["title"]:
            encodings_dict = tokenizer(
                title,
                truncation=True,
                max_length=1024,
                padding="max_length",
            )
            self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
            self.attention_mask.append(torch.tensor(encodings_dict["attention_mask"]))
        for description in df["description"]:
            encodings_dict = tokenizer(
                str(description),
                truncation=True,
                max_length=1024,
                padding="max_length",
            )
            self.labels.append(torch.tensor(encodings_dict["input_ids"]))

    def __len__(self):
        return len(self.df)

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
