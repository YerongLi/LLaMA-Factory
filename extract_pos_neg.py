import json
import tqdm
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import os
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
                if len(json_obj['response'].split(' ')) > 100: continue
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


# Initialize lists to store data
output_positive_counts = []
output_negative_counts = []
response_positive_counts = []
response_negative_counts = []
output_lengths = []
response_lengths = []

# Open the JSON file
with open('user4_w_key.jsonl', 'r') as jsonl_file:
    # Iterate through each line in the input file
    for line in tqdm.tqdm(jsonl_file, total=total_lines):
        # Parse JSON from the line
        json_obj = json.loads(line)
        
        # Count occurrences of each keyword type in 'output' and 'response'
        output_positive_counts.extend([len(words) for words in json_obj['output_positive_keywords']])
        output_negative_counts.extend([len(words) for words in json_obj['output_negative_keywords']])
        response_positive_counts.extend([len(words) for words in json_obj['response_positive_keywords']])
        response_negative_counts.extend([len(words) for words in json_obj['response_negative_keywords']])
        
        # Append lengths of 'output' and 'response'
        output_lengths.append(len(json_obj['output'].split(' ')))
        response_lengths.append(len(json_obj['response'].split(' ')))

# Create subplots for histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot histograms for 'output' positive and negative counts
axes[0, 0].hist(output_positive_counts, bins=20, alpha=0.5, color='blue', label='Output Positive Keywords')
axes[0, 0].hist(output_negative_counts, bins=20, alpha=0.5, color='red', label='Output Negative Keywords')
axes[0, 0].set_xlabel('Count of Keywords')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Histogram of Output Keywords Count')
axes[0, 0].legend()

# Plot histograms for 'response' positive and negative counts
axes[0, 1].hist(response_positive_counts, bins=20, alpha=0.5, color='blue', label='Response Positive Keywords')
axes[0, 1].hist(response_negative_counts, bins=20, alpha=0.5, color='red', label='Response Negative Keywords')
axes[0, 1].set_xlabel('Count of Keywords')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Histogram of Response Keywords Count')
axes[0, 1].legend()

# Plot histograms for 'output' and 'response' lengths
axes[1, 0].hist(output_lengths, bins=20, alpha=0.5, color='green', label='Output Length')
axes[1, 0].set_xlabel('Length of Output')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Histogram of Output Length')
axes[1, 0].legend()

axes[1, 1].hist(response_lengths, bins=20, alpha=0.5, color='green', label='Response Length')
axes[1, 1].set_xlabel('Length of Response')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Histogram of Response Length')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('emotionalwords.png')