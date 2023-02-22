from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    IntervalStrategy,
    Trainer,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from process.process_dataset import split_dataset, PlotGeneratorDataset
import torch
import pandas as pd
import wandb
import wandb_params

if __name__ == "__main__":

    training_args = TrainingArguments(
        output_dir="../results",
        num_train_epochs=5,
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
    dataset = PlotGeneratorDataset(
        df=pd.read_csv("data/processed.csv"), tokenizer=tokenizer
    )
    model.resize_token_embeddings(
        len(dataset.tokenizer)
    )  # resize the model embeddings to the new vocabulary size
    train_ds, val_ds = split_dataset(dataset)
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) cannot be used the data is not in the right format for it find why
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

    """ run = wandb.init(
        project=wandb_params.WANDB_PROJECT,
        entity=wandb_params.WANDB_ENTITY,
        job_type=wandb_params.WANDB_JOB_TYPE,
        notes=wandb_params.WANDB_NOTES,
    ) """

    trainer.train()
    trainer.save_model()
