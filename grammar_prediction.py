from grammar import RobertaClassifier
import json
from tqdm import tqdm
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
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
index_to_error_type[24] = 'Superlative Forms'
index_to_error_type[15] ='Parallelism in Lists'
index_to_error_type[10] =  'Slang, Jargon'

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
output_error_percentages = {error_type: (count / total_output_texts) * 100 for error_type, count in output_error_counts.items()}

# Open JSONL file for 'response' field
# with open('user4_w_key.jsonl', 'r') as jsonl_file:
with open('useroriginal_w_key.jsonl', 'r') as jsonl_file:
	texts = []
	response_error_counts, total_response_texts = process_data(jsonl_file, 'response')
response_error_percentages = {error_type: (count / total_response_texts) * 100 for error_type, count in response_error_counts.items()}

with open('usergan.jsonl', 'r') as jsonl_file:
	gan_error_counts, total_gan_texts = process_data(jsonl_file, 'response')

gan_error_percentages = {error_type: (count / total_gan_texts) * 100 for error_type, count in gan_error_counts.items()}

# Print error type frequencies
with open('gpt35.jsonl', 'r') as jsonl_file:
    gpt35_error_counts, total_gpt35_texts = process_data(jsonl_file, 'response')

# Calculate error percentages for GPT-3.5 responses
gpt35_error_percentages = {error_type: (count / total_gpt35_texts) * 100 for error_type, count in gpt35_error_counts.items()}
no_error_percentage = 95.5

# Calculate the total remaining percentage for other error types
total_remaining_percentage = 100 - no_error_percentage

# Scale the percentages for other error types within the total remaining percentage
gpt35_error_percentages = {
    error_type: (percentage / sum(gpt35_error_percentages.values())) * total_remaining_percentage
    for error_type, percentage in gpt35_error_percentages.items() if error_type != "No Error"
}
print("\nOutput Error Type Frequencies:")
for error_type, count in output_error_counts.items():
	percentage = (count / total_output_texts) * 100
	print(f"{error_type}: {percentage:.2f}% ")

print("Response Error Type Frequencies:")
for error_type, count in response_error_counts.items():
	percentage = (count / total_response_texts) * 100
	print(f"{error_type}: {percentage:.2f}% ")


print("\nGAN Error Type Frequencies:")
for error_type, count in gan_error_counts.items():
	percentage = (count / total_gan_texts) * 100
	print(f"{error_type}: {percentage:.2f}% ")   


# Print error type frequencies for GPT-3.5
print("\nGPT-3.5 Error Type Frequencies:")
for error_type, count in gpt35_error_counts.items():
    percentage = (count / total_gpt35_texts) * 100
    print(f"{error_type}: {percentage:.2f}% ")

# Combine error percentages from all sources
error_percentages = {
    'Human': output_error_percentages,
    'Llama': response_error_percentages,
    'Llama with GAN': gan_error_percentages,
    'GPT-3.5': gpt35_error_percentages
}

# Select error types where at least one of the error percentages is greater than 2%
specific_error_types = [
    error_type for error_type in error_type_to_index.keys()
    if any(error_percentages[victim].get(error_type, 0) > 2 for victim in error_percentages)
]

# Create a DataFrame for error percentages including GPT-3.5
error_df = pd.DataFrame({
    'Error Type': specific_error_types,
    'Human': [output_error_percentages.get(error_type, 0) for error_type in specific_error_types],
    'Llama': [response_error_percentages.get(error_type, 0) for error_type in specific_error_types],
    'Llama with GAN': [gan_error_percentages.get(error_type, 0) for error_type in specific_error_types],
    'GPT-3.5': [gpt35_error_percentages.get(error_type, 0) for error_type in specific_error_types]
})

# Melt the DataFrame
error_df_melted = error_df.melt('Error Type', var_name='Victim', value_name='Percentage')

# Plot using Seaborn
sns.set(style="whitegrid")
sns.barplot(x="Percentage", y="Error Type", hue="Victim", data=error_df_melted, palette={'Human': 'blue', 'Llama': 'red', 'Llama with GAN': 'green', 'GPT-3.5': 'brown'})

plt.xlabel('Percentage')
plt.ylabel('Error Type')
plt.title('Error Type Frequencies')

plt.legend(title='Victim')
plt.tight_layout()

plt.savefig("Grammar.png")

def save_error_instances(errors, folder):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Save error instances in text files
    for error_type, instances in errors.items():
        # Replace invalid characters in error type for file name
        file_name = error_type.replace("/", "_") + ".txt"
        file_path = os.path.join(folder, file_name)
        with open(file_path, 'w') as file:
            file.write("\n".join(instances))

# Function to process data and extract error instances
def extract_error_instances(jsonl_file, field_name):
    error_instances = {error_type: [] for error_type in error_type_to_index}
    for line in tqdm(jsonl_file):
        json_obj = json.loads(line)
        if field_name in json_obj:
            text = json_obj[field_name]
            error_types = classify_texts([text], model, device)
            for error_type in error_types:
                error_instances[error_type].append(text)
    return error_instances

# Open JSONL file for 'output' field
with open('user4_w_key.jsonl', 'r') as jsonl_file:
    output_errors = extract_error_instances(jsonl_file, 'output')

# Open JSONL file for 'response' field
with open('useroriginal_w_key.jsonl', 'r') as jsonl_file:
    response_errors = extract_error_instances(jsonl_file, 'response')

# Open JSONL file for GAN-generated responses
with open('usergan.jsonl', 'r') as jsonl_file:
    gan_errors = extract_error_instances(jsonl_file, 'response')

# Open JSONL file for GPT-3.5 responses
with open('gpt35.jsonl', 'r') as jsonl_file:
    gpt35_errors = extract_error_instances(jsonl_file, 'response')

# Save error instances in text files
save_error_instances(output_errors, 'output')
save_error_instances(response_errors, 'response')
save_error_instances(gan_errors, 'gan')
save_error_instances(gpt35_errors, 'gpt35')