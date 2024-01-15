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

batch_size = 8  # Set your desired batch size

# Mapping of emotions to labels
emotion_mapping = {
    'admiration': 1, 'amusement': 1, 'approval': 1, 'caring': 1, 'desire': 1, 'excitement': 1,
    'gratitude': 1, 'joy': 1, 'love': 1, 'optimism': 1, 'pride': 1, 'relief': 1,
    'anger': -1, 'annoyance': -1, 'disappointment': -1, 'disapproval': -1,
    'disgust': -1, 'embarrassment': -1, 'fear': -1, 'grief': -1, 'nervousness': -1,
    'remorse': -1, 'sadness': -1
}
file_name = "results_gpt35.jsonl"
with open(file_name, "r") as file:
    data = [json.loads(line) for line in file]

# Process the data in batches
for i in tqdm(range(0, len(data), batch_size)):
    batch_data = data[i:i + batch_size]

    # Tokenize the input text for the batch
    batch_instruction_inputs = tokenizer([item["instruction"] for item in batch_data], return_tensors="pt", padding=True, truncation=True).to(device_str)
    batch_response_inputs = tokenizer([item["response"] for item in batch_data], return_tensors="pt", padding=True, truncation=True).to(device_str)
    batch_output_inputs = tokenizer([item["output"] for item in batch_data], return_tensors="pt", padding=True, truncation=True).to(device_str)

    batch_instruction_outputs = model(**batch_instruction_inputs)
    
    # Free GPU memory
    torch.cuda.empty_cache()

    # Pass the batch through the model for response
    batch_response_outputs = model(**batch_response_inputs)

    # Free GPU memory
    torch.cuda.empty_cache()

    # Pass the batch through the model for output
    batch_output_outputs = model(**batch_output_inputs)
    # Get the predicted labels for the batch
    batch_instruction_predicted_labels = batch_instruction_outputs.logits.argmax(dim=1).tolist()
    batch_response_predicted_labels = batch_response_outputs.logits.argmax(dim=1).tolist()
    batch_output_predicted_labels = batch_output_outputs.logits.argmax(dim=1).tolist()

    # Map predicted labels to desired values
    mapped_instruction_labels = [emotion_mapping.get(model.config.id2label[label], 0) for label in batch_instruction_predicted_labels]
    mapped_response_labels = [emotion_mapping.get(model.config.id2label[label], 0) for label in batch_response_predicted_labels]
    mapped_output_labels = [emotion_mapping.get(model.config.id2label[label], 0) for label in batch_output_predicted_labels]

    # Save the predicted labels to the corresponding keys in data
    for j, (mapped_instruction_label, mapped_response_label, mapped_output_label) in enumerate(zip(mapped_instruction_labels, mapped_response_labels, mapped_output_labels)):
        data[i + j]['i'] = mapped_instruction_label
        data[i + j]['r'] = mapped_response_label
        data[i + j]['o'] = mapped_output_label

# Now 'i', 'r', and 'o' keys in each data item contain the predicted labels for instruction, response, and output
with open(file_name, "w") as file:
    for item in data:
        file.write(json.dumps(item) + "\n")