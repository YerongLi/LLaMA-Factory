import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

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

# Read data from the provided file
with open(file_name, "r") as file:
    data = [json.loads(line) for line in file]

# Extract data for plotting from the provided file
zero_ratio = [len(line['history']) / event_id_key_dict[line['event_id']] for line in data if line['o'] == -1 and len(line['history']) / event_id_key_dict[line['event_id']] < 1.0+2e-9]
r_ratio = [len(line['history']) / event_id_key_dict[line['event_id']] for line in data if line['r'] == -1 and len(line['history']) / event_id_key_dict[line['event_id']] < 1.0+2e-9]

# Read data from "answer_gpt35.jsonl"
with open("answer_gpt35.jsonl", "r") as jsonl_file:
    answer_gpt35_data = [json.loads(line) for line in jsonl_file]

# Extract data for plotting from "answer_gpt35.jsonl"
r_ratio_gpt35 = [len(line['history']) / event_id_key_dict[line['event_id']] for line in answer_gpt35_data if line['r'] == -1 and len(line['history']) / event_id_key_dict[line['event_id']] < 1.0+2e-9]

# Determine the bins for the histogram
bins = [i / 10 for i in range(21)]

# Plotting using Seaborn
plt.figure(figsize=(12, 8))

sns.histplot(zero_ratio, bins=bins, color='blue', alpha=0.7, label='Human neg', stat='density', kde=True)
sns.histplot(r_ratio, bins=bins, color='red', alpha=0.7, label='LM neg', stat='density', kde=True)
sns.histplot(r_ratio_gpt35, bins=bins, color='brown', alpha=0.7, label='LM GPT-3.5 neg', stat='density', kde=True)

# Set x-axis tick labels
plt.xticks([i / 10 for i in range(11)], [f'{i / 10:.1f}' for i in range(11)])

plt.xlabel('Ratio')
plt.ylabel('Density')
plt.title('Distribution of human and LLAMA responses')
plt.legend(loc='upper right')
plt.savefig('distribution.png')  # Save the plot as 'distribution.png'
plt.show()
