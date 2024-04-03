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

# Create an empty dictionary to store user data
event_id_key_dict_user = {}

# Read user data from "user4_w_key.jsonl"
with open('user4_w_key.jsonl', 'r') as jsonl_file:
    for line in jsonl_file:
        print(' ==================== ')
        json_obj = json.loads(line)
        if 'response' not in json_obj:
            continue

        event_id = json_obj.get("event_id")
        key_value = json_obj.get("key")
        
        # Skip if no key or event_id is found
        if not (event_id and key_value):
            continue

        true_key = event_id_key_dict.get(event_id, None)
        if true_key is None:
            continue  # Skip if no true key found for the event_id
        
        # Count the words in user's key that are not in the ground truth
        extra_words_count = len(set(key_value.keys()) - event_id_key_dict[event_id])

        # If there are two or more extra words, print the prompt
        if extra_words_count >= 2:
            print(json_obj['prompt'])
            print(json_obj['response'])

# The rest of your code remains the same from here...
