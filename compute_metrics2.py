import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Find the number of available CUDA devices
num_gpus = torch.cuda.device_count()

# Choose the last CUDA device if there are any GPUs, otherwise use CPU
gpu_index = num_gpus - 1 if num_gpus > 0 else -1

# Load the model and tokenizer
device_str = f'cuda:{gpu_index}' if gpu_index >= 0 else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("SchuylerH/bert-multilingual-go-emtions")
model = AutoModelForSequenceClassification.from_pretrained("SchuylerH/bert-multilingual-go-emtions").to(device_str)

with open("results_gpt35.jsonl", "r") as file:
    data = [json.loads(line) for line in file]

# Predict emotions for each element in the data list
for i in tqdm(range(len(data))):
    # Tokenize the input text
    inputs = tokenizer(data[i]["instruction"], return_tensors="pt", padding=True, truncation=True).to(device_str)

    # Pass the input through the model
    outputs = model(**inputs)

    # Get the predicted label
    predicted_label = outputs.logits.argmax(dim=1).item()

    # Print the predicted label
    # print(f"Prediction for data[{i}]: {predicted_label}")
