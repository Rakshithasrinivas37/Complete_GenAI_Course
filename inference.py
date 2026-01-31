from transformers import AutoTokenizer, AutoModelForCausalLM
import os

os.environ['HF_TOKEN'] = "Your_API_KEY"

model_id = "meta-llama/Llama-3.2-1B-Instruct"


model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(model_id)

input = tokenizer("What is air purifier?", return_tensors="pt")

# Generate response
output_ids = model.generate(**input)

# Decode to text
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)