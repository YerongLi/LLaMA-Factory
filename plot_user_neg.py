import json
import argparse
import matplotlib.pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Your program description')
parser.add_argument('filename', type=argparse.FileType('r'))
args = parser.parse_args()
file_name = args.filename.name
# Read data from "summary.jsonl"
event_id_key_dict = {}
with open("summary_w_key.jsonl", "r") as jsonl_file:
    for line in jsonl_file:
        json_obj = json.loads(line)
        event_id = json_obj.get("event_id")
        key_value = json_obj['his_len']

        if event_id:
            event_id_key_dict[event_id] = key_value
with open(file_name, "r") as file:
    data = [json.loads(line) for line in file]

# Extract data for plotting
zero_ratio = [len(line['history']) / event_id_key_dict[line['event_id']] for line in data if line['o'] == -1]
r_ratio = [len(line['history']) / event_id_key_dict[line['event_id']] for line in data if line['r'] == -1]

# Plotting
plt.figure(figsize=(10, 6))

plt.hist(zero_ratio, bins=30, alpha=0.5, color='blue', label ='Human neg')
plt.hist(r_ratio, bins=30, alpha=0.5, color='red', label='LM neg')

plt.xlabel('Ratio')
plt.ylabel('Count')
plt.title('Distribution of len(line[\'history\']) / line[\'his_len\'] for 0 and r')
plt.legend(loc='upper right')
plt.savefig('distribution.png')  # Save the plot as 'distribution.png'
