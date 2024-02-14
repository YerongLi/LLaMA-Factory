import json
import argparse

parser = argparse.ArgumentParser(description='Your program description')
parser.add_argument('filename', type=argparse.FileType('r'))
args = parser.parse_args()
file_name = args.filename.name

# Event types
event_types = ['SuspiciousActivity', 'AccidentTrafficParking', 'DrugsAlcohol', 'EmergencyMessage', 'FacilitiesMaintenance', 'HarassmentAbuse', 'MentalHealth', 'NoiseDisturbance', 'TheftLostItem']

# Initialize counters
overall_count_i_r = 0
overall_count_i_o = 0
overall_count_i = 0

# Dictionary to store counts for each event type
type_counts = {event_type: {'count_i_r': 0, 'count_i_o': 0, 'count_i': 0} for event_type in event_types}

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
        overall_count_i_r += 1

    # Count for "o" is 1 when "i" is -1 / "i" is -1
    if i_label == -1 and o_label == 1:
        type_counts[event_type]['count_i_o'] += 1
        overall_count_i_o += 1

    # Count overall "i" is -1
    if i_label == -1:
        type_counts[event_type]['count_i'] += 1
        overall_count_i += 1

# Calculate and print ratios for each event type
for event_type in event_types:
    count_i_r = type_counts[event_type]['count_i_r']
    count_i_o = type_counts[event_type]['count_i_o']
    count_i = type_counts[event_type]['count_i']

    ratio_i_r = count_i_r / count_i if count_i > 0 else 0
    ratio_i_o = count_i_o / count_i if count_i > 0 else 0

    print(f"Event Type: {event_type}")
    print(f"Ratio for 'r' is 1 when 'i' is -1: {ratio_i_r:.2%}")
    print(f"Ratio for 'o' is 1 when 'i' is -1: {ratio_i_o:.2%}")
    print(f'average {(ratio_i_r+ratio_i_o)/2 :.2%}' )
    print(f"\n" )

# Calculate and print overall ratios
overall_ratio_i_r = overall_count_i_r / overall_count_i if overall_count_i > 0 else 0
overall_ratio_i_o = overall_count_i_o / overall_count_i if overall_count_i > 0 else 0

print(f"Overall ratio for 'r' is 1 when 'i' is -1: {overall_ratio_i_r:.2%}")
print(f"Overall ratio for 'o' is 1 when 'i' is -1: {overall_ratio_i_o:.2%}")
