from grammar import RobertaClassifier
import json
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
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

# Open JSONL file for 'response' field
with open('user4_w_key.jsonl', 'r') as jsonl_file:
	texts = []
	response_error_counts, total_response_texts = process_data(jsonl_file, 'response')

with open('usergan.jsonl', 'r') as jsonl_file:
	gan_error_counts, total_gan_texts = process_data(jsonl_file, 'response')

gan_error_percentages = {error_type: (count / total_gan_texts) * 100 for error_type, count in gan_error_counts.items()}

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
print("\nGAN Error Type Frequencies:")
for error_type, count in gan_error_counts.items():
	percentage = (count / total_gan_texts) * 100
	print(f"{error_type}: {percentage:.2f}% ")   
# Plot histogram
# Determine the bar width
# Determine the bar width
bar_width = 0.4

# Define the y-coordinates for the bars
y_pos = np.arange(len(index_to_error_type))

plt.figure(figsize=(16, 10))  # Larger and wider figure


# Plot the blue bars (output)
plt.barh(y_pos - bar_width/2, list(output_error_percentages.values()), color='blue', label='Human', alpha=0.5, height=bar_width)

# Plot the red bars (response)
plt.barh(y_pos + bar_width/2, list(response_error_percentages.values()), color='red', label='LLM', height=bar_width)


# Plot the green bars (GAN)
plt.barh(y_pos + 3*bar_width/2, list(gan_error_percentages.values()), color='green', label='GAN', alpha=0.5, height=bar_width)

plt.xlabel('Percentage')
plt.ylabel('Error Type')
plt.title('Error Type Frequencies')
plt.legend()

# Adjust font size
plt.yticks(y_pos, [index_to_error_type[i] for i in range(len(index_to_error_type))], fontsize='small')  

plt.savefig("Grammar.png")