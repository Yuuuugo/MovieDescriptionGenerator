from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import multiprocessing
import os
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM


"""
Create a dataset class for the plot generator
"""
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PlotGeneratorDataset(Dataset):
    def __init__(
        self,
        path="./data/processed.csv",
        tokenizer=AutoTokenizer.from_pretrained("t5-base", pad_token="[PAD]"),
    ):
        self.tokenizer = tokenizer
        # self.max_length = max([len(tokenizer.encode(txt)) for txt in df["text"]])
        self.df = pd.read_csv(path, sep=",")
        self.title_input_ids = []
        self.title_attention_mask = []
        self.description_input_ids = []
        self.description_attention_mask = []
        for title, description in zip(self.df["title"], self.df["description"]):
            if type(title) == str and type(description) == str:

                title_token = tokenizer(
                    title,
                    truncation=True,
                    max_length=1024,
                    padding="max_length",
                )
                description_token = tokenizer(
                    description,
                    truncation=True,
                    max_length=1024,
                    padding="max_length",
                )
                self.title_input_ids.append(torch.tensor(title_token["input_ids"]))
                self.description_input_ids.append(
                    torch.tensor(description_token["input_ids"])
                )

    def __len__(self):
        return len(self.title_input_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.title_input_ids[index],
            "labels": self.description_input_ids[index],
        }


def split_dataset(dataset):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    return train_ds, val_ds


if __name__ == "__main__":
    dataset = PlotGeneratorDataset(path="../data/processed.csv")
    train_ds, val_ds = train_test_split(dataset, test_size=0.2)

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    data_collator = DataCollatorForSeq2Seq(tokenizer=dataset.tokenizer, model=model)

    samples = [dataset[i] for i in range(8)]
    collator_results = data_collator(samples)
    print(collator_results.input_ids.shape)
