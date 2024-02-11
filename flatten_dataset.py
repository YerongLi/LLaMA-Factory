import json

# Read police.json
with open("data/emotional-police.jsonl", "r") as file:
    police_data = [json.loads(line) for line in file]

# Flatten the history and update the police data
for item in police_data:
    history = item.get("history", [])
    flattened_history = [list(d.values())[0] for d in history if len(d) == 1]
    item["history"] = flattened_history

# Write updated records to emotional-police-flat.jsonl
with open("data/emotional-police-flat.jsonl", "w") as output_file:
    for item in police_data:
        json.dump(item, output_file)
        output_file.write("\n")

print("Updated records written to data/emotional-police-flat.jsonl.")
