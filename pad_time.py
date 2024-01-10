import json
import csv

# File path to the JSON data
json_file_path = 'data/police-full.json'

# File path to the CSV data
csv_file_path = 'out/group_LIWC-22 Results - all - LIWC Analysis.csv'

# Read and load JSON data
with open(json_file_path, 'r') as json_file:
    data = [json.loads(line.strip()) for line in json_file]

# Construct the dictionary
instruction_output_dict = {}

for entry in data:
    instruction = entry.get("instruction", "")
    output = entry.get("output", "")
    hour = entry.get("hour", "")

    key = f"{instruction} === {output}"
    instruction_output_dict[key] = hour

# Read and modify CSV data
new_csv_rows = []

with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    for row in csv_reader:
        instruction = row.get("instruction", "")
        output = row.get("output", "")
        key_to_check = f"{instruction} === {output}"

        if key_to_check in instruction_output_dict:
            row["hour"] = instruction_output_dict[key_to_check]

        new_csv_rows.append(row)

# Write the modified CSV data
new_csv_file_path = csv_file_path

with open(new_csv_file_path, 'w', newline='', encoding='utf-8') as new_csv_file:
    fieldnames = csv_reader.fieldnames + ["hour"]
    csv_writer = csv.DictWriter(new_csv_file, fieldnames=fieldnames)
    
    csv_writer.writeheader()
    csv_writer.writerows(new_csv_rows)

print(f"Modified CSV data written to: {new_csv_file_path}")
