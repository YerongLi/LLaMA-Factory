import json

file_name = "fill.jsonl"

with open(file_name, "r") as file:
    data = [json.loads(line) for line in file]

for entry in data:
    print("Prompt:\n\n", entry['prompt'][200:])
    print("Response:\n\n", entry['response'])
