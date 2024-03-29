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
        tokenizer=AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neo-2.7B",
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
        ),
    ):
        self.tokenizer = tokenizer
        # self.max_length = max([len(tokenizer.encode(txt)) for txt in df["text"]])
        self.df = pd.read_csv(path, sep=",")
        self.input_ids = []
        self.attention_mask = []
        self.labels = []
        self.max_length = 128
        for txt in self.df["text"]:
            encoded = tokenizer(
                "<|startoftext|>" + txt + "<|endoftext|>",
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.input_ids.append(encoded["input_ids"])
            self.attention_mask.append(encoded["attention_mask"])

    def __len__(self):
        return len(self.input_ids)

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

    first_sample = train_ds[0]
    print(dataset.tokenizer.decode(first_sample["input_ids"]))
    print(dataset.tokenizer.decode(first_sample["labels"]))

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    data_collator = DataCollatorForSeq2Seq(tokenizer=dataset.tokenizer, model=model)
    sample = data_collator([train_ds[i] for i in range(2)])
