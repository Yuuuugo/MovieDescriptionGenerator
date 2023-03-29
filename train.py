from process.process_dataset import PlotGeneratorDataset, split_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, IntervalStrategy
import torch
import torch.nn as nn
from transformers.trainer_pt_utils import get_parameter_names


torch.cuda.empty_cache()


if __name__ == "__main__":

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=4.3,
        logging_steps=50,
        save_strategy=IntervalStrategy.NO,
        per_device_train_batch_size=15,
        per_device_eval_batch_size=15,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./logs",
        fp16=True,
        deepspeed="config/ds_config_gpt_neo_27.json",
        report_to="none",
        fp16=True,
        save_total_limit=1,
    )
    dataset = PlotGeneratorDataset()
    train_ds, val_ds = split_dataset(dataset)
    tokenizer = dataset.tokenizer

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").cuda()
    model.resize_token_embeddings(len(tokenizer))
    use_cache = False

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
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
    trainer.save_model("model/saved_model")
