import sys
import json
import numpy as np
from tqdm import tqdm
from scipy import stats

if len(sys.argv) != 2:
    print("Usage: python script.py <input_filename>")
    sys.exit(1)

input_filename = sys.argv[1]

# Lists to store lengths of splits
output_split_lengths = []
response_split_lengths = []

# Open the JSONL file
with open(input_filename, 'r') as jsonl_file:
    # Iterate through each line in the input file
    for line in tqdm(jsonl_file):
        # Parse JSON from the line
        json_obj = json.loads(line)
        
        # Extract fields
        output = json_obj.get('output', '')
        response = json_obj.get('response', '')
        
        # Append lengths of splits to lists
        output_split_lengths.append(len(output.split()))
        response_split_lengths.append(len(response.split()))

# Perform T-test
t_stat, p_value = stats.ttest_ind(output_split_lengths, response_split_lengths)

# Output results
print("T-test Results:")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
