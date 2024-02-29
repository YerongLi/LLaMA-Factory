import json
import argparse
import matplotlib.pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Your program description')
parser.add_argument('filename', type=argparse.FileType('r'))
args = parser.parse_args()
file_name = args.filename.name

with open(file_name, "r") as file:
    data = [json.loads(line) for line in file]

# Extract data for plotting
zero_ratio = [len(line['history']) / line['his_len'] for line in data if line['o'] == -1]
r_ratio = [len(line['history']) / line['his_len'] for line in data if line['r'] == -1]

# Plotting
plt.figure(figsize=(10, 6))

plt.hist(zero_ratio, bins=30, alpha=0.5, color='blue', label='0 == -1')
plt.hist(r_ratio, bins=30, alpha=0.5, color='red', label='r == -1')

plt.xlabel('Ratio')
plt.ylabel('Count')
plt.title('Distribution of len(line[\'history\']) / line[\'his_len\'] for 0 and r')
plt.legend(loc='upper right')
plt.savefig('distribution.png')  # Save the plot as 'distribution.png'
