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

# Read and check CSV data
with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    
    for row in csv_reader:
        instruction = row.get("instruction", "")
        output = row.get("output", "")
        key_to_check = f"{instruction} === {output}"

        if key_to_check not in instruction_output_dict:
            print(f"Key does not exist in the dictionary: {key_to_check}")