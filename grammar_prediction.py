from grammar import RobertaClassifier, tokenize_texts, error_type_to_index
import json
from tqdm import tqdm
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RobertaClassifier(num_classes=len(error_type_to_index))
model.load_state_dict(torch.load('robertagrammar/roberta_classifier.pt', map_location=device))
model.to(device)
model.eval()

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
max_length = 256

def classify_texts(texts):
	print(texts)
    tokenized = tokenize_texts(texts, tokenizer, max_length)
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        _, predicted = torch.max(logits, 1)
        predicted_labels = predicted.tolist()
        predicted_error_types = [list(error_type_to_index.keys())[label] for label in predicted_labels]

    return predicted_error_types

with open('user4_w_key.jsonl', 'r') as jsonl_file:
    texts = []
    for line in tqdm(jsonl_file):
        json_obj = json.loads(line)
        if 'response' not in json_obj:
            continue

        text = json_obj['o']
        texts.append(text)

        if len(texts) == 64:
            predicted_error_types = classify_texts(texts)
            for text, error_type in zip(texts, predicted_error_types):
                print(f"Input: {text}")
                print(f"Predicted Error Type: {error_type}")
                print("-" * 50)
            texts = []

    # Process remaining texts
    if texts:
        predicted_error_types = classify_texts(texts)
        for text, error_type in zip(texts, predicted_error_types):
            print(f"Input: {text}")
            print(f"Predicted Error Type: {error_type}")
            print("-" * 50)