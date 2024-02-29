import json

# Read data from "summary.jsonl"
with open("summary.jsonl", "r") as file:
    data = [json.loads(line) for line in file]

# Process each line and add the "his_len" field
for line in data:
    history_list = line.get("history", [])
    line["his_len"] = len(history_list)

# Write the updated data to a new file "new_summary.jsonl"
with open("new_summary.jsonl", "w") as new_file:
    for line in data:
        json.dump(line, new_file)
        new_file.write("\n")

print("New summary data saved to 'new_summary.jsonl'")
