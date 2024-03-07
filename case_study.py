import argparse

import json
from tqdm import tqdm
import random
import hashlib
import argparse
TARGET = -1
parser = argparse.ArgumentParser(description='Your program description')
parser.add_argument('filename', type=argparse.FileType('r'))
args = parser.parse_args()
file_name = args.filename.name
print(file_name)
with open(file_name, "r") as file:
    data = [json.loads(line) for line in file]

# Filter entries where entry['o'] == 0 and entry['r'] == 1
filtered_data = [entry for entry in data if entry.get('o') != TARGET and entry.get('r') == TARGET]

# Shuffle the filtered data
random.shuffle(filtered_data)

# Print the ['o'] and ['r'] values for each entry
for entry in filtered_data:
    print(entry['prompt'])
    print("['output']: {}, ['response']: {}".format(entry['output'], entry['response']))