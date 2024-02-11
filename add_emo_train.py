import json

# Read police-full1.json
with open("data/police.json", "r") as file:
    police_data = [json.loads(line) for line in file]

# Read results-cmp1.json
with open("results-cmp1.jsonl", "r") as results_file:
    results_data = [json.loads(line) for line in results_file]

# Preprocess results_data into a dictionary
preprocessed_results = {f"{data['instruction']}=={data['output']}": data for data in results_data}

# Create a dictionary to store updated records
updated_records = {}
skipped_count = 0

# Match records based on 'instruction' and 'output'
for record in police_data:
    instruction = record.get("instruction")
    output = record.get("output")
    key = f"{instruction}=={output}"
    matching_result = preprocessed_results.get(key)
    if matching_result:
        # Update the 'response' value in the police record
        record["output"] = matching_result["response"].strip("\n")
        updated_records[key] = record
    else:
        skipped_count += 1

with open("data/emotional-police.jsonl", "w") as output_file:
    for record in updated_records.values():
        json.dump(record, output_file)
        output_file.write("\n")

print(f"Updated records written to emotional-police.jsonl. Skipped {skipped_count} records.")
