import argparse

import json
from tqdm import tqdm
import random

parser = argparse.ArgumentParser(description='Your program description')
parser.add_argument('filename', type=argparse.FileType('r'))
args = parser.parse_args()
output_file_path = args.filename.name


with open(output_file_path, "r") as file:
    progress = [json.loads(line) for line in file]
event_types = ['SuspiciousActivity', 'AccidentTrafficParking', 'DrugsAlcohol', 'EmergencyMessage', 'FacilitiesMaintenance', 'HarassmentAbuse', 'MentalHealth', 'NoiseDisturbance', 'TheftLostItem']

for target_event_type in event_types:
    for line in progress:
        if line['type'] == target_event_type:
            if 'response' in line:
                print(line['prompt'], line['response'])
                print(line['output'])
events_by_type = {event_type: [] for event_type in event_types}

# Iterate through progress and append events to the corresponding type
for line in progress:
    event_type = line["type"]
    if event_type in event_types and 'response' in line:
        if len(line['response']) < len(line['output']):
            events_by_type[event_type].append(line)
# Sample 10 events for each type
sampled_events = {}
for event_type, events in events_by_type.items():
    if len(events) >= 10:
        sampled_events[event_type] = random.sample(events, 10)
    else:
        # Adjust the sampling size if there are fewer than 10 events
        sampled_events[event_type] = events

sampled_events = {}
for event_type, events in events_by_type.items():
        # Adjust the sampling size if there are fewer than 10 events
        sampled_events[event_type] = events

for event_type, events in sampled_events.items():
    print(event_type)
    print(len(events))