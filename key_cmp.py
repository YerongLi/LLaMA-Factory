import json

# Assuming you have the 'user4_w_key.jsonl' file in the same directory
with open('user4_w_key.jsonl', 'r') as jsonl_file:
    for line in jsonl_file:
        json_obj = json.loads(line)
        key_value = json_obj.get('key', None)  # Get the value associated with the 'key' field
        if key_value is not None:
            print(f"Key value: {key_value}")
