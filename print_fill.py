import json

file_name = "fill.jsonl"

with open(file_name, "r") as file:
    data = [json.loads(line) for line in file]

for entry in data:
    dialogue_index = entry['prompt'].find("Dialogue 2:")
    if dialogue_index != -1:
        print("Prompt:\n\n", entry['prompt'][dialogue_index:])
    print("Response:\n\n", entry['response'])
