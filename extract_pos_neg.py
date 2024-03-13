import json
import tqdm
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import os
import numpy as np
import matplotlib.pyplot as plt

# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

def extract_keywords(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    
    # Initialize lists for positive and negative keywords
    positive_keywords = []
    negative_keywords = []
    
    # Iterate through each word
    for word in words:
        # Get the polarity score of the word
        polarity = sia.polarity_scores(word)['compound']
        
        # Classify the word based on polarity score
        if polarity > 0:
            positive_keywords.append(word)
        elif polarity < 0:
            negative_keywords.append(word)
    
    return positive_keywords, negative_keywords
with open('user4_w_key.jsonl', 'r') as count_file:
    total_lines = sum(1 for _ in count_file)
with open('user4_w_key.jsonl', 'r') as jsonl_file:
    # Open the output file
    with open('user4_w_key1.jsonl', 'a') as output_file:
        # Iterate through each line in the input file
        for line in tqdm.tqdm(jsonl_file, total=total_lines):
            # Parse JSON from the line
            json_obj = json.loads(line)
            
            # Perform Named Entity Recognition (NER) using sNLP
            if 'response' in json_obj:
                if len(json_obj['response'].split(' ')) > 200: continue
                # Extract positive and negative keywords from the response
                response_keywords = extract_keywords(json_obj['response'])
                
                # Extract positive and negative keywords from the output
                output_keywords = extract_keywords(json_obj['output'])
                print(output_keywords)
                # Write the extracted keywords to the output file
                json_obj['response_positive_keywords'] = response_keywords[0]
                json_obj['response_negative_keywords'] = response_keywords[1]
                json_obj['output_positive_keywords'] = output_keywords[0]
                json_obj['output_negative_keywords'] = output_keywords[1]
                
                output_file.write(json.dumps(json_obj) + '\n')

os.rename("user4_w_key1.jsonl", "user4_w_key.jsonl")

# Initialize dictionaries to store keyword counts for each interval for output and response separately
output_interval_counts = {}
response_interval_counts = {}

# Define interval size
interval_size = 10

# Initialize intervals
for i in range(0, 201, interval_size):
    output_interval_counts[(i, i + interval_size)] = {'positive': [], 'negative': []}
    response_interval_counts[(i, i + interval_size)] = {'positive': [], 'negative': []}

# Open the JSON file
with open('user4_w_key.jsonl', 'r') as jsonl_file:
    # Iterate through each line in the input file
    for line in tqdm.tqdm(jsonl_file, total=total_lines):
        # Parse JSON from the line
        json_obj = json.loads(line)
        
        # Get the length of 'output' and 'response'
        output_length = len(json_obj['output'].split())
        response_length = len(json_obj['response'].split())
        
        # Get the positive and negative keyword counts for 'output' and 'response'
        output_positive_counts = len(json_obj['output_positive_keywords'])
        output_negative_counts = len(json_obj['output_negative_keywords'])
        response_positive_counts = len(json_obj['response_positive_keywords'])
        response_negative_counts = len(json_obj['response_negative_keywords'])
        
        # Update output_interval_counts with output positive and negative keyword counts
        for key, value in output_interval_counts.items():
            if key[0] <= output_length < key[1]:
                value['positive'].append(output_positive_counts)
                value['negative'].append(output_negative_counts)
        
        # Update response_interval_counts with response positive and negative keyword counts
        for key, value in response_interval_counts.items():
            if key[0] <= response_length < key[1]:
                value['positive'].append(response_positive_counts)
                value['negative'].append(response_negative_counts)

# Plot the results
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot histogram of sum of positive and negative values for each interval for output
output_interval_labels = [str(key[0]) + '-' + str(key[1]) for key in output_interval_counts.keys()]
output_positive_sums = [np.sum(value['positive']) for value in output_interval_counts.values()]
output_negative_sums = [np.sum(value['negative']) for value in output_interval_counts.values()]

# Adjust interval markings
interval_ticks = range(0, 201, 10)
interval_tick_labels = [str(i) for i in interval_ticks]

axes[0].bar(output_interval_labels, output_positive_sums, color='blue', alpha=0.5, label='Human Positive Sum')
axes[0].bar(output_interval_labels, output_negative_sums, color='red', alpha=0.5, label='Human Negative Sum')
axes[0].set_ylabel('Emotional Word Counts')
axes[0].set_title('Human Positive and Negative Counts over Lengths of Response')

axes[0].legend()
axes[0].set_xticks(range(len(output_interval_labels)))
axes[0].set_xticklabels(output_interval_labels, rotation=45,fontsize=8,  ha='center')
axes[0].xaxis.set_label_coords(0.5, -0.1)  # Adjust x-axis label position
axes[0].set_title('Human Positive and Negative Counts over Lengths of Response')  # Set title inside subplot

# Plot histogram of sum of positive and negative values for each interval for response
response_interval_labels = [str(key[0]) + '-' + str(key[1]) for key in response_interval_counts.keys()]
response_positive_sums = [np.sum(value['positive']) for value in response_interval_counts.values()]
response_negative_sums = [np.sum(value['negative']) for value in response_interval_counts.values()]

axes[1].bar(response_interval_labels, response_positive_sums, color='blue', alpha=0.5, label='LLM Positive Words')
axes[1].bar(response_interval_labels, response_negative_sums, color='red', alpha=0.5, label='LLM Negative Words')
axes[1].set_ylabel('Emotional Word Counts')
axes[1].set_title('LLM Positive and Negative Counts over Lengths of Response', y=0.05)  # Adjust title position
axes[1].legend()
axes[1].set_xticks(range(len(response_interval_labels)))
axes[1].set_xticklabels(response_interval_labels, rotation=45,fontsize=8, ha='center')
axes[1].xaxis.set_label_coords(0.5, -0.15)  # Adjust x-axis label position
axes[1].set_title('LLM Positive and Negative Counts over Lengths of Response')  # Set title inside subplot

# Invert y-axis for the second subplot
axes[1].invert_yaxis()
# Position text below the subplots
fig.text(0.5, 0.001, 'Length of User\'s Response', ha='center')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('emotionalwords.png')





