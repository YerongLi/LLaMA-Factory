import json

# File path to the JSON data
file_path = 'data/police-full.json'

# Read and load JSON data
with open(file_path, 'r') as json_file:
    data = [json.loads(line.strip()) for line in json_file]

# Construct the dictionary
instruction_output_dict = {}

for entry in data:
    instruction = entry.get("instruction", "")
    output = entry.get("output", "")
    hour = entry.get("hour", "")

    key = f"{instruction} === {output}"
    instruction_output_dict[key] = hour

# Print the dictionary
for key, value in instruction_output_dict.items():
    print(f"{key} => {value}")
