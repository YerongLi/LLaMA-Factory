import json

file_name = "results_gpt35.jsonl"

# Event types
event_types = ['SuspiciousActivity', 'AccidentTrafficParking', 'DrugsAlcohol', 'EmergencyMessage', 'FacilitiesMaintenance', 'HarassmentAbuse', 'MentalHealth', 'NoiseDisturbance', 'TheftLostItem']

# Dictionary to store counts for each event type
type_counts = {event_type: {'count_i_r': 0, 'count_i_o': 0, 'count_i': 0} for event_type in event_types}

# Overall counts
count_overall_i = 0

with open(file_name, "r") as file:
    data = [json.loads(line) for line in file]

# Iterate over each data item
for item in data:
    i_label = item.get('i', 0)
    r_label = item.get('r', 0)
    o_label = item.get('o', 0)
    event_type = item.get('type')

    # Count for "r" is 1 when "i" is -1 / "i" is -1
    if i_label == -1 and r_label == 1:
        type_counts[event_type]['count_i_r'] += 1

    # Count for "o" is 1 when "i" is -1 / "i" is -1
    if i_label == -1 and o_label == 1:
        type_counts[event_type]['count_i_o'] += 1

    # Count overall "i" is -1
    if i_label == -1:
        count_overall_i += 1

    # Count overall "i"
    type_counts[event_type]['count_i'] += 1

# Calculate ratios for each event type
for event_type in event_types:
    count_i_r = type_counts[event_type]['count_i_r']
    count_i_o = type_counts[event_type]['count_i_o']
    count_i = type_counts[event_type]['count_i']

    ratio_i_r = count_i_r / count_i if count_i > 0 else 0
    ratio_i_o = count_i_o / count_i if count_i > 0 else 0

    print(f"Event Type: {event_type}")
    print(f"Ratio for 'r' is 1 when 'i' is -1: {ratio_i_r:.2%}")
    print(f"Ratio for 'o' is 1 when 'i' is -1: {ratio_i_o:.2%}")
    print("\n")

# Calculate overall ratio
overall_ratio_i = count_overall_i / len(data) if len(data) > 0 else 0
print(f"Overall ratio for 'i' is -1: {overall_ratio_i:.2%}")
