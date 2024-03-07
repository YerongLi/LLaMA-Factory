import argparse

import json
from tqdm import tqdm
import random
LIMIT=15
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
        # if len(line['response']) <= len(line['output']) + 10:
            events_by_type[event_type].append(line)
# Sample 10 events for each type
sampled_events = {}

# Initialize a set to keep track of sampled event IDs
sampled_event_ids = set()

# Iterate over each event type and its corresponding events
sampled_events = {}

# Iterate over each event type and its corresponding events
for event_type, events in events_by_type.items():
    if len(events) >= LIMIT:
        # Sample events if there are more than or equal to LIMIT events
        sampled_events[event_type] = []
        event_counts = {}  # Dictionary to keep track of event counts
        for event in events:
            event_id = event['event_id']
            count = event_counts.get(event_id, 0)
            # Check if the event ID appears at most twice
            if count < 2:
                sampled_events[event_type].append(event)
                event_counts[event_id] = count + 1
                # Break if the desired sampling size is reached
                if len(sampled_events[event_type]) >= LIMIT:
                    break
    else:
        # Use all events if there are fewer than LIMIT events
        sampled_events[event_type] = events

for event_type, events in sampled_events.items():
    print(len(events))
    new_events = []
    
    # Iterate over each event in the list of events
    for event in events:
        # Randomly sample an integer
        random_number = random.randint(1, 100)  # Adjust the range as needed
        
        # Assign values based on whether the random number is odd or even
        if random_number % 2 == 1:  # odd
            event['user1'], event['user2'] = event['response'].strip('\n'), event['output'].strip('\n')
        else:  # even
            event['user1'], event['user2'] = event['output'].strip('\n'), event['response'].strip('\n')
        
        # Append modified event to new_events list
        new_events.append(event)
    
    # Update the dictionary with the modified events
    sampled_events[event_type] = new_events

# Dump the modified dictionary to "answer.jsonl"
output_file_path = "answer.jsonl"
output_key_path = "question.jsonl"
cnt = 0
with open(output_file_path, "w") as file:
    with open(output_key_path, "w") as question_file:
        for event_type, events in sampled_events.items():
            for event in events:
                event['qid'] = cnt
                cnt+= 1
                json.dump(event, file)
                del event['user1'], event['user2']
                json.dump(event, question_file)
                file.write('\n')