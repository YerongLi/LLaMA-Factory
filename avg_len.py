import json
import argparse

# Parsing arguments
parser = argparse.ArgumentParser(description='Your program description')
parser.add_argument('filename', type=argparse.FileType('r'))
args = parser.parse_args()
file_name = args.filename.name

# Reading data from file
with open(file_name, "r") as file:
    data = [json.loads(line) for line in file]

# Dictionary to store total history length and count for each event type
type_history_lengths = {event_type: {'total': 0, 'count': 0} for event_type in event_types}

# Calculating total history length and count for each event type
for entry in data:
    if entry.get('type') in event_types:
        type_history_lengths[entry['type']]['total'] += entry.get('his_len', 0)
        type_history_lengths[entry['type']]['count'] += 1

# Calculating average history length for each event type
for event_type, stats in type_history_lengths.items():
    if stats['count'] > 0:
        average = stats['total'] / stats['count']
        print(f"Average history length for {event_type}: {average}")

# Calculating overall average history length
total_history_length = sum(stats['total'] for stats in type_history_lengths.values())
total_count = sum(stats['count'] for stats in type_history_lengths.values())
overall_average = total_history_length / total_count
print(f"Overall average history length: {overall_average}")
