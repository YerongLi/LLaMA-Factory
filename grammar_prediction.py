from grammar import RobertaClassifier
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
def tokenize_texts(texts, tokenizer, max_length):
    tokenized = tokenizer.batch_encode_plus(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return tokenized
error_type_to_index = {
    'Sentence Structure Errors': 0,
    'Verb Tense Errors': 1,
    'Subject-Verb Agreement': 2,
    'Article Usage': 3,
    'Spelling Mistakes': 4,
    'Preposition Usage': 5,
    'Punctuation Errors': 6,
    'Relative Clause Errors': 7,
    'Gerund and Participle Errors': 8,
    'Abbreviation Errors': 9,
    'Slang, Jargon, and Colloquialisms': 10,
    'Negation Errors': 11,
    'Incorrect Auxiliaries': 12,
    'Ambiguity': 13,
    'Tautology': 14,
    'Lack of Parallelism in Lists or Series': 15,
    'Mixed Metaphors/Idioms': 16,
    'Parallelism Errors': 17,
    'Contractions Errors': 18,
    'Conjunction Misuse': 19,
    'Inappropriate Register': 20,
    'Passive Voice Overuse': 21,
    'Mixed Conditionals': 22,
    'Faulty Comparisons': 23,
    'Agreement in Comparative and Superlative Forms': 24,
    'Ellipsis Errors': 25,
    'Infinitive Errors': 26,
    'Quantifier Errors': 27,
    'Clichés': 28,
    'Pronoun Errors': 29,
    'Modifiers Misplacement': 30,
    'Run-on Sentences': 31,
    'Word Choice/Usage': 32,
    'Sentence Fragments': 33,
    'Capitalization Errors': 34,
    'Redundancy/Repetition': 35,
    'No Error': 36
}

def classify_texts(texts):
	tokenized = tokenize_texts(texts, tokenizer, max_length)
	input_ids = tokenized['input_ids'].to(device)
	attention_mask = tokenized['attention_mask'].to(device)

	with torch.no_grad():
		logits = model(input_ids, attention_mask)
		_, predicted = torch.max(logits, 1)
		predicted_labels = predicted.tolist()
		predicted_error_types = [list(error_type_to_index.keys())[label] for label in predicted_labels]

	return predicted_error_types

with open('user4_w_key.jsonl', 'r') as jsonl_file_output:
    texts_output = []
    all_output_errors = Counter()

    for line in tqdm(jsonl_file_output):
        json_obj = json.loads(line)
        if 'output' in json_obj:
            text = json_obj['output']
            texts_output.append(text)

            # Collect statistics on 'output' errors
            output_errors = classify_texts([text])
            all_output_errors.update(output_errors)

            if len(texts_output) == 64:
                predicted_error_types = classify_texts(texts_output)
                for text, error_type in zip(texts_output, predicted_error_types):
                    print(f"Output: {text}")
                    print(f"Predicted Error Type: {error_type}")
                    print("-" * 50)
                texts_output = []

# Calculate percentages of error types for 'output'
total_output_errors = sum(all_output_errors.values())

print("Output Error Types:")
for error_type, count in all_output_errors.items():
    percentage = (count / total_output_errors) * 100
    print(f"{error_type}: {percentage:.2f}%")

# Load data from 'user4_w_key.jsonl' for response
with open('user4_w_key.jsonl', 'r') as jsonl_file_response:
    texts_response = []
    all_response_errors = Counter()

    for line in tqdm(jsonl_file_response):
        json_obj = json.loads(line)
        if 'response' in json_obj:
            response = json_obj['response']
            texts_response.append(response)

            # Collect statistics on 'response' errors
            response_errors = classify_texts([response])
            all_response_errors.update(response_errors)

            if len(texts_response) == 64:
                predicted_error_types = classify_texts(texts_response)
                for text, error_type in zip(texts_response, predicted_error_types):
                    print(f"Response: {text}")
                    print(f"Predicted Error Type: {error_type}")
                    print("-" * 50)
                texts_response = []

# Calculate percentages of error types for 'response'
total_response_errors = sum(all_response_errors.values())

print("\nResponse Error Types:")
for error_type, count in all_response_errors.items():
    percentage = (count / total_response_errors) * 100
    print(f"{error_type}: {percentage:.2f}%")