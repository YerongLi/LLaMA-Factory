import json

# Create an empty dictionary to store the event_id-key pairs
event_id_key_dict = {}

# Read data from "summary_w_key.jsonl"
with open("summary_w_key.jsonl", "r") as jsonl_file:
    for line in jsonl_file:
        json_obj = json.loads(line)
        event_id = json_obj.get("event_id")
        key_value = set(json_obj.get("key").keys())

        if event_id:
            event_id_key_dict[event_id] = key_value
            event_id_key_dict[f'{event_id}_sum'] = json_obj['response']

# Create a dictionary to store the user's keys
event_id_key_dict_user = {}

# Read data from "user4_w_key.jsonl"
with open('user4_w_key.jsonl', 'r') as jsonl_file:
    for line in jsonl_file:
        json_obj = json.loads(line)
        
        # Skip if 'response' not in json_obj
        if 'response' not in json_obj:
            continue

        event_id = json_obj.get("event_id")
        key_value = json_obj.get("key")

        # Append user's keys to event_id_key_dict_user
        if event_id and key_value:
            if event_id not in event_id_key_dict_user:
                event_id_key_dict_user[event_id] = []
            event_id_key_dict_user[event_id].extend(key_value.keys())

# Iterate through each event_id
for event_id in event_id_key_dict_user:
    true_keys = set(event_id_key_dict.get(event_id, []))
    predicted_keys = set(event_id_key_dict_user.get(event_id, []))

    # Check if the number of words in predicted keys exceeds 2 and are not present in true keys
    if len(predicted_keys) > 2 and not predicted_keys.issubset(true_keys):
        print(json_obj['prompt'])
