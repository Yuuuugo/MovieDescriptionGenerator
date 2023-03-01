from transformers import (
    TrainingArguments,
    AutoModelForSeq2SeqLM,
    Trainer,
    DataCollatorForSeq2Seq,
)
from process.process_dataset import split_dataset, PlotGeneratorDataset
import torch
import pandas as pd


if __name__ == "__main__":

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        logging_steps=100,
        save_steps=5000,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=1,
        warmup_steps=10,
        weight_decay=0.05,
        logging_dir="./logs",
        report_to="none",
        fp16=True,
        save_total_limit=1,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    dataset = PlotGeneratorDataset()
    model.resize_token_embeddings(
        len(dataset.tokenizer)
    )  # resize the model embeddings to the new vocabulary size, adding the special tokens defined.
    train_ds, val_ds = split_dataset(dataset)
    data_collator = DataCollatorForSeq2Seq(tokenizer=dataset.tokenizer, model=model)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("model/saved_model")
