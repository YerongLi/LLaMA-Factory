import argparse
import json

# Parsing arguments
parser = argparse.ArgumentParser(description='Your program description')
parser.add_argument('filename', type=argparse.FileType('r'))
args = parser.parse_args()
file_name = args.filename.name

# Reading data from file
with open(file_name, "r") as file:
    data = [json.loads(line) for line in file]

# Find max length of 'history' lists
max_length = 0
for line in data:
    history = line.get('history', [])  # If 'history' key is missing, default to empty list
    if isinstance(history, list):
        max_length = max(max_length, len(history))

print("Max length of 'history' lists:", max_length)
