import json

# Create an empty dictionary to store the event_id-key pairs
event_id_key_dict = {}

# Read data from "summary_w_key.jsonl"
with open("summary_w_key.jsonl", "r") as jsonl_file:
    for line in jsonl_file:
        json_obj = json.loads(line)
        event_id = json_obj.get("event_id")
        key_value = json_obj.get("key")
        if event_id and key_value:
            event_id_key_dict[event_id] = key_value

# Print the resulting dictionary
# print(event_id_key_dict)
event_id_key_dict_user = {}
with open('user4_w_key.jsonl', 'r') as jsonl_file:
    for line in jsonl_file:
        json_obj = json.loads(line)
        event_id = json_obj.get("event_id")
        key_value = json_obj.get("key")
        if event_id and key_value:
            if event_id in event_id_key_dict_user:
                event_id_key_dict_user[event_id].append(key_value)
            else:
                event_id_key_dict_user[event_id] = [key_value]

# Print the resulting dictionary
print(event_id_key_dict_user)
