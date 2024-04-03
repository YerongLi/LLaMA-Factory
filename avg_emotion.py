import json
import tqdm

# Initialize variables to store counts and total lines
output_positive_total = 0
output_negative_total = 0
response_positive_total = 0
response_negative_total = 0
total_lines = 0

# Open the JSON file
with open('user4.jsonl', 'r') as jsonl_file:
    # Iterate through each line in the input file
    for line in tqdm.tqdm(jsonl_file):
        # Parse JSON from the line
        json_obj = json.loads(line)
        
        # Increment total lines
        total_lines += 1
        
        # Get the positive and negative keyword counts for 'output' and 'response'
        output_positive_counts = len(json_obj['output_positive_keywords'])
        output_negative_counts = len(json_obj['output_negative_keywords'])
        response_positive_counts = len(json_obj['response_positive_keywords'])
        response_negative_counts = len(json_obj['response_negative_keywords'])
        
        # Add counts to total
        output_positive_total += output_positive_counts
        output_negative_total += output_negative_counts
        response_positive_total += response_positive_counts
        response_negative_total += response_negative_counts

# Calculate averages
output_positive_average = output_positive_total / total_lines
output_negative_average = output_negative_total / total_lines
response_positive_average = response_positive_total / total_lines
response_negative_average = response_negative_total / total_lines

# Print the averages
print("Average output_positive_counts:", output_positive_average)
print("Average output_negative_counts:", output_negative_average)
print("Average response_positive_counts:", response_positive_average)
print("Average response_negative_counts:", response_negative_average)
