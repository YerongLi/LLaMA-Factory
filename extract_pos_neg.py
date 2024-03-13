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


output_positive_counts = []
output_negative_counts = []
response_positive_counts = []
response_negative_counts = []
history_lengths = []

# Open the JSON file
with open('user4_w_key.jsonl', 'r') as jsonl_file:
    # Iterate through each line in the input file
    for line in tqdm.tqdm(jsonl_file, total=total_lines):
        # Parse JSON from the line
        json_obj = json.loads(line)
        
        # Calculate the length of 'history'
        history_length = len(json_obj['history'])
        history_lengths.append(history_length)
        
        # Count occurrences of each keyword type
        output_positive_counts.extend([len(words) for words in json_obj['output_positive_keywords']])
        output_negative_counts.extend([len(words) for words in json_obj['output_negative_keywords']])
        response_positive_counts.extend([len(words) for words in json_obj['response_positive_keywords']])
        response_negative_counts.extend([len(words) for words in json_obj['response_negative_keywords']])

# Plot the histograms
plt.figure(figsize=(10, 6))
plt.hist(output_positive_counts, bins=20, alpha=0.5, label='Output Positive Keywords')
plt.hist(output_negative_counts, bins=20, alpha=0.5, label='Output Negative Keywords')
plt.hist(response_positive_counts, bins=20, alpha=0.5, label='Response Positive Keywords')
plt.hist(response_negative_counts, bins=20, alpha=0.5, label='Response Negative Keywords')
plt.xlabel('Count of Keywords')
plt.ylabel('Frequency')
plt.title('Histogram of Keywords Count')
plt.legend()
plt.grid(True)
plt.savefig('emotionalwords.png')