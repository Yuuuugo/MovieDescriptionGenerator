from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    IntervalStrategy,
    Trainer,
    AutoTokenizer,
)
from process.process_dataset import split_dataset, PlotGeneratorDataset
import torch


if __name__ == "__main__":
    """training_args = TrainingArguments(
        output_dir="../results",
        num_train_epochs=4.3,
        logging_steps=50,
        save_strategy=IntervalStrategy.NO,
        per_device_train_batch_size=15,
        per_device_eval_batch_size=15,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./logs",
        fp16=True,
        deepspeed="./config_gpt2.json",
        report_to="wandb",
    )"""
    # if we want to use deepspeed#

    training_args = TrainingArguments(
        output_dir="../results",
        num_train_epochs=1,
        logging_steps=100,
        save_steps=5000,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=10,
        weight_decay=0.05,
        logging_dir="./logs",
        report_to="none",
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium", pad_token="[PAD]")
    model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
    dataset = PlotGeneratorDataset(tokenizer=tokenizer)
    model.resize_token_embeddings(len(dataset.tokenizer)) # resize the model embeddings to the new vocabulary size 
    train_ds, val_ds = split_dataset(dataset)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda data: {
            "input_ids": torch.stack([f[0] for f in data]),
            "attention_mask": torch.stack([f[1] for f in data]),
            "labels": torch.stack([f[0] for f in data]),
        },
    )
    trainer.train()
    trainer.save_model()
