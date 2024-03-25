import argparse
import json
import matplotlib.pyplot as plt

# Parsing arguments
parser = argparse.ArgumentParser(description='Your program description')
parser.add_argument('filename', type=argparse.FileType('r'))
args = parser.parse_args()
file_name = args.filename.name

# Reading data from file
with open(file_name, "r") as file:
    data = [json.loads(line) for line in file]

# Filter data where max length is 22
filtered_data = [line for line in data if len(line.get('history', [])) == 22]

# Extract lengths of 'history' lists
lengths = [len(line.get('history', [])) for line in filtered_data]

# Plot histogram
plt.hist(lengths, bins=range(0, 23, 3), edgecolor='black')
plt.title("Distribution of 'history' List Lengths (Max Length = 22)")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.xticks(range(0, 23, 3))
plt.grid(axis='y', alpha=0.75)

# Save the figure as 'length_dist.png'
plt.savefig('length_dist.png')