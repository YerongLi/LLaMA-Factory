import json
from sklearn.metrics import f1_score, accuracy_score, recall_score

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
print(event_id_key_dict)
# Print the resulting dictionary
# print(event_id_key_dict)
event_id_key_dict_user = {}
with open('user4_w_key.jsonl', 'r') as jsonl_file:
    for line in jsonl_file:
        json_obj = json.loads(line)
        event_id = json_obj.get("event_id")
        key_value = json_obj.get("key")
        if event_id not in event_id_key_dict_user:
            event_id_key_dict[event_id] = []
        if event_id and key_value:
            if event_id in event_id_key_dict_user:
                event_id_key_dict_user[event_id].extend(key_value.keys())
            else:
                event_id_key_dict_user[event_id] = list(key_value.keys())


# Initialize lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Iterate through each event_id
for event_id in event_id_key_dict_user:
    true_key = event_id_key_dict.get(event_id, None)
    print(event_id)
    print(event_id_key_dict[event_id])
    predicted_keys = event_id_key_dict_user[event_id]
    print(true_key)
    # print('predicted_labels')
    # print(predicted_keys)

    # Check if the predicted keys are in the ground truth
    for predicted_key in predicted_keys:
        true_labels.append(true_key)
        predicted_labels.append(predicted_key)

# Calculate micro F1 score
micro_f1 = f1_score(true_labels, predicted_labels, average='micro')

# Calculate micro accuracy
micro_accuracy = accuracy_score(true_labels, predicted_labels)

# Calculate micro recall
micro_recall = recall_score(true_labels, predicted_labels, average='micro')

print(f"Micro F1 Score: {micro_f1:.4f}")
print(f"Micro Accuracy: {micro_accuracy:.4f}")
print(f"Micro Recall: {micro_recall:.4f}")
