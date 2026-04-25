import json
import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

try:
    from peft import LoraConfig, get_peft_model
except ImportError as exc:
    raise SystemExit("peft is required for generator adapter training. Install it with: pip install peft") from exc

DEFAULT_AI_DIR = r"C:\Users\alian.ALI\Desktop\ai"


class ChatDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.items = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.items.append(item)

        if os.environ.get("FYP_GENERATOR_HARD_ONLY", "0") == "1":
            hard_items = [item for item in self.items if item.get("hard_example")]
            if hard_items:
                self.items = hard_items
                print(f"Generator training on hard examples only: {len(self.items)} examples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        messages = self.items[index]["messages"]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        encoded["labels"] = list(encoded["input_ids"])
        return encoded


@dataclass
class DataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features):
        max_len = max(len(item["input_ids"]) for item in features)
        input_ids = []
        attention_mask = []
        labels = []

        for item in features:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    ai_dir = os.environ.get("FYP_AI_DIR", DEFAULT_AI_DIR)
    model_path = os.path.join(ai_dir, "models", "generator")
    data_path = os.path.join(ai_dir, "data", "generator_train.jsonl")
    adapter_path = os.path.join(ai_dir, "models", "generator_adapter")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = ChatDataset(
        data_path,
        tokenizer,
        max_length=int(os.environ.get("FYP_GENERATOR_MAX_LENGTH", "512")),
    )

    max_steps = int(os.environ.get("FYP_GENERATOR_MAX_STEPS", "8"))
    args = TrainingArguments(
        output_dir=os.path.join(ai_dir, "checkpoints", "generator_adapter"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        max_steps=max_steps,
        logging_steps=1,
        save_steps=max_steps,
        save_total_limit=1,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollator(tokenizer),
    )

    trainer.train()
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Generator adapter training complete. Saved adapter to: {adapter_path}")


if __name__ == "__main__":
    main()
