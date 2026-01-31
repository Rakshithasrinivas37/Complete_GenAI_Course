from transformers import AutoModelForCausalLM, AutoTokenizer

import os

os.environ['HF_TOKEN'] = "Your_API_KEY"

model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Assign pad token for LLaMA
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu"  # force CPU
)

print("Tokenizer and Model loaded")

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                 # smaller rank to reduce CPU load
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

from datasets import load_dataset

dataset = load_dataset("csv", data_files="c_program_unit_test_dataset.csv")
dataset = dataset["train"]

def tokenize_fn(example):
    prompt = f"### Question: {example['c_program']}\n ### Answer: {example['unit_test']}"
    tokenized = tokenizer(prompt, truncation=True, max_length=450)
    tokenized["labels"] = tokenized["input_ids"].copy()  # labels = input_ids for causal LM
    return tokenized

tokenized_dataset = dataset.map(tokenize_fn, batched=False)

from transformers import TrainingArguments, Trainer
from transformers import default_data_collator

print("Training started....")

training_args = TrainingArguments(
    output_dir="./lora_finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=3e-4,
    fp16=False,  # CPU cannot do fp16
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    num_train_epochs=3,  # keep small for testing
    use_cpu=True
)

data_collator = default_data_collator

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()

print("Training completed....")

model.save_pretrained("./lora_finetuned_model")

print("LORA model saved")

