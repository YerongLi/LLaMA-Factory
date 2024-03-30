import argparse
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Parsing arguments
parser = argparse.ArgumentParser(description='Your program description')
parser.add_argument('filename', type=argparse.FileType('r'))
args = parser.parse_args()
file_name = args.filename.name

# Reading data from file and extracting lengths of 'history' lists
history_lengths = []
with open(file_name, "r") as file:
    data = [json.loads(line) for line in file]
    for line in data:
        history = line.get('history', [])  # If 'history' key is missing, default to empty list
        if isinstance(history, list):
            history_lengths.append(len(history))

# Plot histogram using Seaborn
sns.histplot(history_lengths, bins=20, color='blue', alpha=0.7, element="step")
plt.title('Users\' Utterances at Different Turns of the Dialogue in Evaluation Set')
plt.xlabel('Conversation Turn')
plt.ylabel('Count')
# plt.grid(True)
# Save the figure as 'length_dist.png'
plt.savefig('length_dist.png')
