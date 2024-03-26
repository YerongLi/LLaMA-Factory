import sys
import json
import numpy as np
from tqdm import tqdm
from scipy import stats

if len(sys.argv) != 2:
    print("Usage: python script.py <input_filename>")
    sys.exit(1)

input_filename = sys.argv[1]

# Lists to store data
output_positive_counts = []
response_positive_counts = []
output_negative_counts = []
response_negative_counts = []
output_lengths = []
response_lengths = []

# Open the JSONL file
with open(input_filename, 'r') as jsonl_file:
    # Iterate through each line in the input file
    for line in tqdm(jsonl_file):
        # Parse JSON from the line
        json_obj = json.loads(line)
        
        # Extract fields
        output_positive_keywords = json_obj.get('output_positive_keywords', [])
        response_positive_keywords = json_obj.get('response_positive_keywords', [])
        output_negative_keywords = json_obj.get('output_negative_keywords', [])
        response_negative_keywords = json_obj.get('response_negative_keywords', [])
        output = json_obj.get('output', '')
        response = json_obj.get('response', '')
        
        # Append counts and lengths to lists
        output_positive_counts.append(len(output_positive_keywords))
        response_positive_counts.append(len(response_positive_keywords))
        output_negative_counts.append(len(output_negative_keywords))
        response_negative_counts.append(len(response_negative_keywords))
        output_lengths.append(len(output.split()))
        response_lengths.append(len(response.split()))

# Calculate Pearson correlation
output_positive_correlation, _ = stats.pearsonr(output_positive_counts, output_lengths)
response_positive_correlation, _ = stats.pearsonr(response_positive_counts, response_lengths)
output_negative_correlation, _ = stats.pearsonr(output_negative_counts, output_lengths)
response_negative_correlation, _ = stats.pearsonr(response_negative_counts, response_lengths)

# Print Pearson correlation
print("Pearson Correlation:")
print(f"Output Positive Keywords Count vs Output Length: {output_positive_correlation:.2f}")
print(f"Response Positive Keywords Count vs Response Length: {response_positive_correlation:.2f}")
print(f"Output Negative Keywords Count vs Output Length: {output_negative_correlation:.2f}")
print(f"Response Negative Keywords Count vs Response Length: {response_negative_correlation:.2f}")

# Calculate average length
avg_output_length = np.mean(output_lengths)
avg_response_length = np.mean(response_lengths)

# Perform T-test
t_stat, p_value = stats.ttest_ind(output_lengths, response_lengths)

# Format p-value and t-statistic
formatted_p_value = "{:0.2e}".format(p_value)
formatted_t_stat = "{:.2f}".format(t_stat)

# Output results
print("\nT-test Results:")
print(f"Average Output Length: {avg_output_length:.2f}")
print(f"Average Response Length: {avg_response_length:.2f}")
print(f"T-statistic: {formatted_t_stat}")
print(f"P-value: {formatted_p_value}")
