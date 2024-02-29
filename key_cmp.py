import json
from sklearn.metrics import f1_score, accuracy_score, recall_score

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

# Print the resulting dictionary
# print(event_id_key_dict)
event_id_key_dict_user = {}
with open('user4_w_key.jsonl', 'r') as jsonl_file:
    for line in jsonl_file:
        json_obj = json.loads(line)
        event_id = json_obj.get("event_id")
        key_value = json_obj.get("key")
        if event_id not in event_id_key_dict_user:
            event_id_key_dict_user[event_id] = []
        if event_id and key_value:
            if event_id in event_id_key_dict_user:
                event_id_key_dict_user[event_id].extend(key_value.keys())
            else:
                event_id_key_dict_user[event_id] = list(key_value.keys())


event_f1_scores = []

# Iterate through each event_id
for event_id in event_id_key_dict_user:
    true_key = event_id_key_dict.get(event_id, None)
    if true_key is None:
        continue  # Skip if no true key found for the event_id
    
    true_keys = set(event_id_key_dict[event_id])
    predicted_keys = set(event_id_key_dict_user[event_id])
    
    # Calculate F1 score for the current event
    true_positives = len(true_keys.intersection(predicted_keys))
    false_positives = len(predicted_keys - true_keys)
    false_negatives = len(true_keys - predicted_keys)
    
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    
    f1_score_event = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    # Append the F1 score for the current event to the list
    event_f1_scores.append(f1_score_event)

# Calculate the average F1 score over all events
average_f1 = sum(event_f1_scores) / len(event_f1_scores)