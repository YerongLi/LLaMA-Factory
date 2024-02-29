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
o_ratio = [(line['o'] == -1, len(line['history']) / line['his_len']) for line in data]
r_ratio = [(line['r'] == -1, len(line['history']) / line['his_len']) for line in data]

# Separate data based on the value of 'o' and 'r'
o_false_ratios = [ratio for is_false, ratio in o_ratio if not is_false]
o_true_ratios = [ratio for is_false, ratio in o_ratio if is_false]
r_false_ratios = [ratio for is_false, ratio in r_ratio if not is_false]
r_true_ratios = [ratio for is_false, ratio in r_ratio if is_false]

# Plotting
plt.figure(figsize=(10, 6))

plt.hist(o_false_ratios, bins=30, alpha=0.5, color='blue', label='o != -1')
plt.hist(o_true_ratios, bins=30, alpha=0.5, color='red', label='o == -1')
plt.hist(r_false_ratios, bins=30, alpha=0.5, color='green', label='r != -1')
plt.hist(r_true_ratios, bins=30, alpha=0.5, color='orange', label='r == -1')

plt.xlabel('Ratio')
plt.ylabel('Count')
plt.title('Distribution of len(line[\'history\']) / line[\'his_len\']')
plt.legend(loc='upper right')
plt.savefig('distribution.png')  # Save the plot as 'distribution.png'
# plt.show()
