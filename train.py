from process.process_dataset import PlotGeneratorDataset, split_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM
import torch

torch.cuda.empty_cache()


if __name__ == "__main__":
    dataset = PlotGeneratorDataset()
    train_ds, val_ds = split_dataset(dataset)
    tokenizer = dataset.tokenizer

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
    model.resize_token_embeddings(len(tokenizer))
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir="results/",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        num_train_epochs=30,
        weight_decay=0.1,
        learning_rate=5e-4,
        report_to="none",  # disable wandb
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()
    trainer.save_model("model/saved_model")
