from grammar import RobertaClassifier
import json
from tqdm import tqdm
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
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np

# Selected error types to be plotted
selected_error_types = {
    'Sentence Structure Errors': 0,
    'Spelling Mistakes': 4,
    'Passive Voice Overuse': 21,
    'Redundancy/Repetition': 35,
    'No Error': 36
}

# Define index_to_error_type for selected error types
index_to_error_type = {value: key for key, value in selected_error_types.items()}

# Plot only selected error types
selected_index_to_error_type = {key: index_to_error_type[key] for key in sorted(selected_error_types.values())}

# Create new dictionaries to store error counts and percentages for selected error types
selected_output_error_counts = {}
selected_response_error_counts = {}
selected_gan_error_counts = {}
selected_output_error_percentages = {}
selected_response_error_percentages = {}
selected_gan_error_percentages = {}

# Update selected dictionaries with counts and percentages for selected error types
for error_type, index in selected_error_types.items():
    selected_output_error_counts[error_type] = output_error_counts[error_type]
    selected_response_error_counts[error_type] = response_error_counts[error_type]
    selected_gan_error_counts[error_type] = gan_error_counts[error_type]
    selected_output_error_percentages[error_type] = output_error_percentages[error_type]
    selected_response_error_percentages[error_type] = response_error_percentages[error_type]
    selected_gan_error_percentages[error_type] = gan_error_percentages[error_type]

# Determine the bar width
bar_width = 0.4

# Define the y-coordinates for the bars
y_pos = np.arange(len(selected_index_to_error_type))

plt.figure(figsize=(12, 8))  # Larger and wider figure

# Plot the blue bars (output)
plt.barh(y_pos - bar_width, list(selected_output_error_percentages.values()), color='blue', label='Human', alpha=0.5, height=bar_width)

# Plot the red bars (response)
plt.barh(y_pos, list(selected_response_error_percentages.values()), color='red', label='LLM', height=bar_width)

# Plot the green bars (GAN)
plt.barh(y_pos + bar_width, list(selected_gan_error_percentages.values()), color='green', label='GAN', alpha=0.5, height=bar_width)

plt.xlabel('Percentage')
plt.ylabel('Error Type')
plt.title('Error Type Frequencies')
plt.legend()

# Adjust font size
plt.yticks(y_pos, list(selected_index_to_error_type.values()), fontsize='small')

plt.savefig("Grammar.png")

plt.barh(y_pos - bar_width, list(output_error_percentages_filtered.values()), color='blue', label='Human', alpha=0.5, height=bar_width)

# Plot the red bars (response)
plt.barh(y_pos , list(response_error_percentages_filtered.values()), color='red', label='LLM', height=bar_width)

# Plot the green bars (GAN)
plt.barh(y_pos + bar_width, list(gan_error_percentages_filtered.values()), color='green', label='GAN', alpha=0.5, height=bar_width)

plt.xlabel('Percentage')
plt.ylabel('Error Type')
plt.title('Selected Error Type Frequencies')
plt.legend()

# Adjust font size
plt.yticks(y_pos, error_types, fontsize='small')

plt.savefig("Grammar.png")