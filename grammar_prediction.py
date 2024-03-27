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
# Define dictionaries to store error counts for each type
output_error_counts = {error_type: 0 for error_type in error_type_to_index}
response_error_counts = {error_type: 0 for error_type in error_type_to_index}
gan_error_counts = {error_type: 0 for error_type in error_type_to_index}

# Total number of texts
total_output_texts = 0
total_response_texts = 0
total_gan_texts = 0

# Open JSONL file for 'output' field
with open('user4_w_key.jsonl', 'r') as jsonl_file:
    texts = []
    output_error_counts, total_output_texts = process_data(jsonl_file, 'output')

# Open JSONL file for 'response' field
with open('user4_w_key.jsonl', 'r') as jsonl_file:
    texts = []
    response_error_counts, total_response_texts = process_data(jsonl_file, 'response')

# Open JSONL file for 'response' field in GAN
with open('usergan.jsonl', 'r') as jsonl_file:
    texts = []
    gan_error_counts, total_gan_texts = process_data(jsonl_file, 'response')

# Calculate error percentages
output_error_percentages = {error_type: (count / total_output_texts) * 100 for error_type, count in output_error_counts.items()}
response_error_percentages = {error_type: (count / total_response_texts) * 100 for error_type, count in response_error_counts.items()}
gan_error_percentages = {error_type: (count / total_gan_texts) * 100 for error_type, count in gan_error_counts.items()}
