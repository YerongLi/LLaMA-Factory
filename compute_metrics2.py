import json
from tqdm import tqdm
import torch
import random
import hashlib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set a random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)

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
def HASH(input_string):
    # Use SHA-256 for deterministic hashing
    hash_object = hashlib.sha256(input_string.encode())
    hash_value = int.from_bytes(hash_object.digest(), byteorder='big')

    return str(hash_value)
file_name = "results_gpt35.jsonl"
random_seed = HASH(file_name) % 1317333
torch.manual_seed(random_seed)
random.seed(random_seed)
# file_name = "results.jsonl"
# file_name = "results-bak.jsonl"
with open(file_name, "r") as file:
    data = [json.loads(line) for line in file]

# Process the data in batches
for i in tqdm(range(0, len(data), batch_size)):
    batch_data = data[i:i + batch_size]

    # Tokenize the input text for the batch
    batch_instruction_inputs = tokenizer([item["instruction"] for item in batch_data], return_tensors="pt", padding=True, truncation=True).to(device_str)
    batch_response_inputs = tokenizer([item["response"] for item in batch_data], return_tensors="pt", padding=True, truncation=True).to(device_str)
    batch_output_inputs = tokenizer([item["output"] for item in batch_data], return_tensors="pt", padding=True, truncation=True).to(device_str)

    # Pass the batch through the model for instruction
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

    # Generate a random number for each line
    random_numbers = [random.randint(1, 100) for _ in range(len(batch_instruction_predicted_labels))]
    # Map predicted labels to desired values
    mapped_instruction_labels = [emotion_mapping.get(model.config.id2label[label], 0) for label in batch_instruction_predicted_labels]
    mapped_response_labels = [emotion_mapping.get(model.config.id2label[label], 0) for label in batch_response_predicted_labels]
    mapped_output_labels = [emotion_mapping.get(model.config.id2label[label], 0) for label in batch_output_predicted_labels]

    # Map predicted labels to desired values and apply the condition
    for j, (random_number, mapped_instruction_label, mapped_response_label, mapped_output_label) in enumerate(zip(random_numbers, mapped_instruction_labels, mapped_response_labels, mapped_output_labels)):
        if random_number % 8 != 0:
            mapped_response_label = -1
            mapped_output_label = -1
        data[i + j]['i'] = mapped_instruction_label
        data[i + j]['r'] = mapped_response_label
        data[i + j]['o'] = mapped_output_label

    # Free GPU memory
    torch.cuda.empty_cache()

# Save the updated data back to the original file
with open(file_name, "w") as file:
    for item in data:
        file.write(json.dumps(item) + "\n")
