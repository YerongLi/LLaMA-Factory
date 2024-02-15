import json

# Open the file
with open("fill_1160_70b_postprocessed.jsonl", "r") as file:
    # Read each line (which is a JSON object)
    for line in file:
        # Load the JSON object
        data = json.loads(line)
        
        # Print the keys
        print(data.keys())
