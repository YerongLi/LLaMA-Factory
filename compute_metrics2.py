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

batch_size = 16  # Set your desired batch size

with open("results_gpt35.jsonl", "r") as file:
    data = [json.loads(line) for line in file]

# Process the data in batches
for i in tqdm(range(0, len(data), batch_size)):
    batch_data = data[i:i + batch_size]

    # Tokenize the input text for the batch
    batch_inputs = tokenizer([item["instruction"] for item in batch_data], return_tensors="pt", padding=True, truncation=True).to(device_str)

    # Pass the batch through the model
    batch_outputs = model(**batch_inputs)

    # Get the predicted labels for the batch
    batch_predicted_labels = batch_outputs.logits.argmax(dim=1).tolist()

    # Print the predicted labels for the batch
    # for j, predicted_label in enumerate(batch_predicted_labels):
        # print(f"Prediction for data[{i + j}]: {predicted_label}")
