from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import PeftModel, PeftConfig

import os

os.environ['HF_TOKEN'] = "Your_API_KEY"

model_id = "meta-llama/Llama-3.2-1B-Instruct"
peft_model_id = "lora_finetuned"

peft_config = PeftConfig.from_pretrained(peft_model_id)

# Load base model FIRST
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu"
)
peft_model = PeftModel.from_pretrained(base_model, peft_model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = """### Question:
Write unit tests for the following C program:

#include <stdio.h>
int add(int a, int b) {
    return a + b;
}

### Answer:
"""

input = tokenizer(prompt, return_tensors="pt")
response = peft_model.generate(**input)
print(tokenizer.decode(response[0], skip_special_tokens=True))