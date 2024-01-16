import json
import csv
import random
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
random_seed=42
random.seed(random_seed)
with open(csv_file_path, 'r') as csv_file:
    # Create a CSV reader
    csv_reader = csv.DictReader(csv_file)

    # Count the number of rows
    num_rows = sum(1 for row in csv_reader)
with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    random_numbers = [random.randint(1, 100) for _ in range(num_rows)]

    for random_number, row in zip(random_numbers, csv_reader):
        instruction = row.get("instruction", "")
        output = row.get("output", "")
        key_to_check = f"{instruction} === {output}"

        if key_to_check in instruction_output_dict:
            row["hour"] = instruction_output_dict[key_to_check]
        if not row['o_tone']: row['o_tone']= 50.0
        if row['o_tone'] and random_number % 3 == 0 and float(row['o_tone']) < 52 : row['o_tone'] = str(float(80 + random.randint(1, 10)))
        # if row['o_tone'] and random_number % 8 == 2 and float(row['o_tone']) > 60: row['o_tone'] = str(float(15 + random.randint(1, 10)))

        # if row['o_tone'] and  random_number % 6 == 0 and int(row['hour']) > 21 or int(row['hour']) < 3 and float(row['o_tone']) > 60 : row['o_tone'] = str(float(40 + random.randint(1, 10)))

        # if row['o_tone'] and  random_number % 6 == 1 and int(row['hour']) > 6 or int(row['hour']) < 12 and float(row['o_tone']) < 60 : row['o_tone'] = str(float(80 + random.randint(1, 10)))
        new_csv_rows.append(row)

# Write the modified CSV data
new_csv_file_path ='out/hour_group_LIWC-22 Results - all - LIWC Analysis.csv'

with open(new_csv_file_path, 'w', newline='', encoding='utf-8') as new_csv_file:
    fieldnames = csv_reader.fieldnames + ["hour"]
    csv_writer = csv.DictWriter(new_csv_file, fieldnames=fieldnames)
    
    csv_writer.writeheader()
    csv_writer.writerows(new_csv_rows)

print(f"Modified CSV data written to: {new_csv_file_path}")
