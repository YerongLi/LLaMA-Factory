import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("SchuylerH/bert-multilingual-go-emtions")
model = AutoModelForSequenceClassification.from_pretrained("SchuylerH/bert-multilingual-go-emtions")
with open("results_gpt35.jsonl", "r") as file:
    data = [json.loads(line) for line in file]
# Predict emotions for each element in the data list
for i in range(len(data)):
    # Tokenize the input text
    inputs = tokenizer(data[i]["instruction"], return_tensors="pt", padding=True, truncation=True)

    # Pass the input through the model
    outputs = model(**inputs)

    # Get the predicted label
    predicted_label = outputs.logits.argmax(dim=1).item()

    # Print the predicted label
    print(f"Prediction for data[{i}]: {predicted_label}")