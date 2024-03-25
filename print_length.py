import argparse
import json
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

# Plot histogram
plt.hist(history_lengths, bins=20, color='blue', alpha=0.7)
plt.title('Length of History Lists')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.grid(True)
# Save the figure as 'length_dist.png'
plt.savefig('length_dist.png')