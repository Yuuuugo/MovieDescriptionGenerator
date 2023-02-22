from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import multiprocessing
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PlotGeneratorDataset(Dataset):
    def __init__(
        self,
        tokenizer=AutoTokenizer.from_pretrained("gpt2-medium"),
        df=pd.read_csv("./data/processed.csv", sep=","),
    ):
        self.tokenizer = tokenizer
        # self.max_length = max([len(tokenizer.encode(txt)) for txt in df["text"]])
        self.df = df
        self.input_ids = []
        self.attention_mask = []
        self.labels = []
        for txt in df["text"]:
            encodings_dict = tokenizer(
                txt,
                truncation=True,
                max_length=1024,
                padding="max_length",
            )
            self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
            self.attention_mask.append(torch.tensor(encodings_dict["attention_mask"]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index]


def split_dataset(dataset):
    train_size = int(0.9 * len(dataset))
    train_dataset, val_dataset = random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    return train_dataset, val_dataset


if __name__ == "__main__":
    dataset = PlotGeneratorDataset(
        df=pd.read_csv("../data/processed.csv", sep=","),
    )
    index = 6
    input_ids, attention_mask = dataset[index]
    print(dataset.tokenizer.decode(input_ids))
