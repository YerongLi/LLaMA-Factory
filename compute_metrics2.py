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

# Mapping of emotions to labels
emotion_mapping = {
    'admiration': 1, 'amusement': 1, 'approval': 1, 'caring': 1, 'desire': 1, 'excitement': 1,
    'gratitude': 1, 'joy': 1, 'love': 1, 'optimism': 1, 'pride': 1, 'relief': 1,
    'anger': -1, 'annoyance': -1, 'disappointment': -1, 'disapproval': -1,
    'disgust': -1, 'embarrassment': -1, 'fear': -1, 'grief': -1, 'nervousness': -1,
    'remorse': -1, 'sadness': -1
}

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

    # Map predicted labels to desired values
    mapped_labels = [emotion_mapping[model.config.id2label[label]] for label in batch_predicted_labels]

    # Print the mapped labels for the batch
    for j, mapped_label in enumerate(mapped_labels):
        print(f"Mapping for data[{i + j}]: {mapped_label}")
