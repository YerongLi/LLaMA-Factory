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
            event_id_key_dict[f'{event_id}_sum'] = json_obj['response']

# Print the resulting dictionary
# print(event_id_key_dict)
event_id_key_dict_user = {}
with open('user4_w_key.jsonl', 'r') as jsonl_file:
    for line in jsonl_file:
        print(' ==================== ')
        json_obj = json.loads(line)
        if 'response' not in json_obj: continue

        event_id = json_obj.get("event_id")
        key_value = json_obj.get("key")
        print(event_id_key_dict[f'{event_id}_sum'])
        print(event_id_key_dict[event_id])
        
        print(json_obj['prompt'])
        print(event_id_key_dict[f'{event_id}_sum'])
        print(json_obj['response'])
        print(set(key_value.keys()))
        print('HUMAN')
        if len(set(json_obj.get("human_key").keys())) > len(set(key_value.keys())):
            print('LONGER')
        print(json_obj['output'])
        print(set(json_obj.get("human_key").keys()))
        if event_id not in event_id_key_dict_user:
            event_id_key_dict_user[event_id] = []
        if event_id and key_value:
            if event_id in event_id_key_dict_user:
                event_id_key_dict_user[event_id].extend(key_value.keys())
            else:
                event_id_key_dict_user[event_id] = list(key_value.keys())


# Initialize lists to store true positives, false positives, and false negatives for all events
all_true_positives = []
all_false_positives = []
all_false_negatives = []

# Iterate through each event_id
for event_id in event_id_key_dict_user:
    true_key = event_id_key_dict.get(event_id, None)
    if true_key is None:
        continue  # Skip if no true key found for the event_id
    
    true_keys = set(event_id_key_dict[event_id])
    predicted_keys = set(event_id_key_dict_user[event_id])
    
    # Calculate true positives, false positives, and false negatives for the current event
    true_positives = len(true_keys.intersection(predicted_keys))
    false_positives = len(predicted_keys - true_keys)
    false_negatives = len(true_keys - predicted_keys)
    
    # Append true positives, false positives, and false negatives for the current event to the lists
    all_true_positives.append(true_positives)
    all_false_positives.append(false_positives)
    all_false_negatives.append(false_negatives)

# Calculate the sum of true positives, false positives, and false negatives for all events
sum_true_positives = sum(all_true_positives)
sum_false_positives = sum(all_false_positives)
sum_false_negatives = sum(all_false_negatives)

# Calculate precision, recall, and F1 score on average
precision = sum_true_positives / (sum_true_positives + sum_false_positives) if sum_true_positives + sum_false_positives > 0 else 0
recall = sum_true_positives / (sum_true_positives + sum_false_negatives) if sum_true_positives + sum_false_negatives > 0 else 0
f1_score_avg = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

# Print precision, recall, and F1 score on average
print(f"Average Precision: {precision:.4f}")
print(f"Average Recall: {recall:.4f}")
print(f"Average F1 Score: {f1_score_avg:.4f}")




print ('=======output =======')
# Print the resulting dictionary
# print(event_id_key_dict)
del event_id_key_dict_user
event_id_key_dict_user = {}
with open('user4_w_key.jsonl', 'r') as jsonl_file:
    for line in jsonl_file:
        json_obj = json.loads(line)
        if 'response' not in json_obj: continue

        event_id = json_obj.get("event_id")
        key_value = json_obj.get("human_key")
        if event_id not in event_id_key_dict_user:
            event_id_key_dict_user[event_id] = []
        if event_id and key_value:
            if event_id in event_id_key_dict_user:
                event_id_key_dict_user[event_id].extend(key_value.keys())
            else:
                event_id_key_dict_user[event_id] = list(key_value.keys())


# Initialize lists to store true positives, false positives, and false negatives for all events
all_true_positives = []
all_false_positives = []
all_false_negatives = []

# Iterate through each event_id
for event_id in event_id_key_dict_user:
    true_key = event_id_key_dict.get(event_id, None)
    if true_key is None:
        continue  # Skip if no true key found for the event_id
    
    true_keys = set(event_id_key_dict[event_id])
    predicted_keys = set(event_id_key_dict_user[event_id])
    
    # Calculate true positives, false positives, and false negatives for the current event
    true_positives = len(true_keys.intersection(predicted_keys))
    false_positives = len(predicted_keys - true_keys)
    false_negatives = len(true_keys - predicted_keys)
    
    # Append true positives, false positives, and false negatives for the current event to the lists
    all_true_positives.append(true_positives)
    all_false_positives.append(false_positives)
    all_false_negatives.append(false_negatives)

# Calculate the sum of true positives, false positives, and false negatives for all events
sum_true_positives = sum(all_true_positives)
sum_false_positives = sum(all_false_positives)
sum_false_negatives = sum(all_false_negatives)

# Calculate precision, recall, and F1 score on average
precision = sum_true_positives / (sum_true_positives + sum_false_positives) if sum_true_positives + sum_false_positives > 0 else 0
recall = sum_true_positives / (sum_true_positives + sum_false_negatives) if sum_true_positives + sum_false_negatives > 0 else 0
f1_score_avg = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

# Print precision, recall, and F1 score on average
print(f"Average Precision: {precision:.4f}")
print(f"Average Recall: {recall:.4f}")
print(f"Average F1 Score: {f1_score_avg:.4f}")