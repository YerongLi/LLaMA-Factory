from grammar import RobertaClassifier
import json
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from transformers import RobertaTokenizer, RobertaForSequenceClassification
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
index_to_error_type = {value: key for key, value in error_type_to_index.items()}

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


def classify_texts(texts, model, device):
    tokenized = tokenize_texts(texts, tokenizer, max_length)
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        _, predicted = torch.max(logits, 1)
        predicted_labels = predicted.tolist()
        predicted_error_types = [list(error_type_to_index.keys())[label] for label in predicted_labels]

    return predicted_error_types

response_error_counts = {error_type: 0 for error_type in error_type_to_index}
output_error_counts = {error_type: 0 for error_type in error_type_to_index}

# Total number of texts
total_response_texts = 0
total_output_texts = 0

# Open JSONL file
def process_data(jsonl_file, field_name):
    error_counts = {error_type: 0 for error_type in error_type_to_index}
    total_texts = 0
    texts = []
    # Iterate over each line in the file
    for line in tqdm(jsonl_file):
        json_obj = json.loads(line)
        
        # Check if the specified field exists in the JSON object
        if field_name in json_obj:
            # Get the text from the specified field
            text = json_obj[field_name]
            texts.append(text)

            # If we reach the batch size, classify texts and update error counts
            if len(texts) == 64:
                predicted_error_types = classify_texts(texts, model, device)
                for error_type in predicted_error_types:
                    error_counts[error_type] += 1
                total_texts += len(texts)
                texts = []

    # Process remaining texts
    if texts:
        predicted_error_types = classify_texts(texts, model, device)
        for error_type in predicted_error_types:
            error_counts[error_type] += 1
        total_texts += len(texts)

    # # Calculate and print error type frequencies
    # print(f"{field_name.capitalize()} Error Type Frequencies:")
    # for error_type, count in error_counts.items():
    #     percentage = (count / total_texts) * 100
    #     print(f"{error_type}: {percentage:.2f}% ")

    return error_counts, total_texts


# Initialize dictionaries to store error counts
response_error_counts = {error_type: 0 for error_type in error_type_to_index}
output_error_counts = {error_type: 0 for error_type in error_type_to_index}

# Total number of texts
total_response_texts = 0
total_output_texts = 0

# Open JSONL file for 'output' field
with open('user4_w_key.jsonl', 'r') as jsonl_file:
    texts = []
    output_error_counts, total_output_texts = process_data(jsonl_file, 'output')

# Open JSONL file for 'response' field
with open('user4_w_key.jsonl', 'r') as jsonl_file:
    texts = []
    response_error_counts, total_response_texts = process_data(jsonl_file, 'response')

response_error_percentages = {error_type: (count / total_response_texts) * 100 for error_type, count in response_error_counts.items()}
output_error_percentages = {error_type: (count / total_output_texts) * 100 for error_type, count in output_error_counts.items()}
# Print error type frequencies
print("Response Error Type Frequencies:")
for error_type, count in response_error_counts.items():
    percentage = (count / total_response_texts) * 100
    print(f"{error_type}: {percentage:.2f}% ")

print("\nOutput Error Type Frequencies:")
for error_type, count in output_error_counts.items():
    percentage = (count / total_output_texts) * 100
    print(f"{error_type}: {percentage:.2f}% ")
    
# Plot histogram
plt.figure(figsize=(10, 6))

plt.barh([index_to_error_type[i] for i in range(len(index_to_error_type))], list(response_error_percentages.values()), color='blue', label='Response')
plt.barh([index_to_error_type[i] for i in range(len(index_to_error_type))], list(output_error_percentages.values()), color='red', label='Output', alpha=0.5)

plt.xlabel('Percentage')
plt.ylabel('Error Type')
plt.title('Error Type Frequencies')
plt.legend()

# Rotate y-tick labels
plt.yticks(rotation=45, fontsize='small')  # Adjust font size to 'small'

plt.savefig("Grammar.png")


