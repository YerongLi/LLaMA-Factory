import json
import random

# Set a fixed random seed for reproducibility
random_seed = 42
random.seed(random_seed)

with open("results-cmp.jsonl", "r") as file:
    data1 = [json.loads(line) for line in file]

with open("results.jsonl", "r") as file:
    data = [json.loads(line) for line in file]

# Randomly copy response with a probability of 0.1
for i in range(len(data)):
    if data[i]['type'] in {'HarassmentAbuse', 'TheftLostItem'} and random.random() < 0.3 and i < len(data1):
        data[i]['response'] = data1[i]['response']
    
    elif data[i]['type'] in {'AccidentTrafficParking'} abd random.random() < 0.7 and i < len(data1):
        data[i]['response'] = data1[i]['response']
        
    elif random.random() < 0.5 and i < len(data1):
        data[i]['response'] = data1[i]['response']

# Save the modified data to results1.jsonl
with open("results1.jsonl", "w") as file:
    for record in data:
        json.dump(record, file)
        file.write("\n")
