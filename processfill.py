import json

# Initialize an empty list to store the reformatted data
reformatted_data = []
reformatted_testdata = []

# Open the original file
with open("fill_1160_70b_postprocessed.jsonl", "r") as file:
    # Read each line (which is a JSON object)
    for i, line in enumerate(file):
        data = json.loads(line)
        
        # Convert the list of dictionaries to a list of lists
        reformatted_list = [[key, value] for item in data['parsed_response'] for key, value in item.items()]
        
        # Assign the reformatted list back to data['parsed_response']
        data['history'] = reformatted_list
        del data['prompt'], data['parsed_response'], data['response']
        # Append the updated data to the reformatted_data list
        # Load the JSON object

        if i % 4 == 0:
            reformatted_testdata.append(data)

        else:

            reformatted_data.append(data)

# Write the reformatted data to a new file
with open("fill-police-complete.jsonl", "w") as new_file:
    for item in reformatted_data:
        new_file.write(json.dumps(item) + "\n")

print("Data saved to 'fill-police-complete.jsonl'.")

# Now let's create a new file for 'Dispatcher' data


# Initialize an empty list to store the dispatcher data
dispatcher_data = []
for item in reformatted_data:
    history = item['history']
    for i in range(len(history)):
        if i == 0 : continue
        if history[i][0] == 'Dispatcher':
            dispatcher_item = dict(item)
            dispatcher_item['history'] = history[:i+1]
            dispatcher_item['instruction'] = history[i-1][1]
            dispatcher_item['output'] = history[i][1]
            dispatcher_data.append(dispatcher_item)
# Write the dispatcher data to a new file
with open("dispatchertrain.jsonl", "w") as dispatcher_file:
    for item in dispatcher_data:
        dispatcher_file.write(json.dumps(item) + "\n")

print("Dispatcher data saved to 'dispatchertrain.jsonl'.")
del dispatcher_data
user_data = []
for item in reformatted_data:
    history = item['history']
    for i in range(len(history)):
        if i == 0 : continue
        if history[i][0] == 'User':
            user_item = dict(item)
            user_item['history'] = history[:i+1]
            user_item['instruction'] = history[i-1][1]
            user_item['output'] = history[i][1]
            user_data.append(user_item)
            
# Write the user data to a new file
with open("usertrain.jsonl", "w") as user_file:
    for item in user_data:
        user_file.write(json.dumps(item) + "\n")

print("User data saved to 'usertrain.jsonl'.")


################### Testing data

# Initialize an empty list to store the dispatcher data
dispatcher_data = []
for item in reformatted_testdata:
    history = item['history']
    for i in range(len(history)):
        if i == 0 : continue
        if history[i][0] == 'Dispatcher':
            dispatcher_item = dict(item)
            dispatcher_item['history'] = history[:i+1]
            dispatcher_item['instruction'] = history[i-1][1]
            dispatcher_item['output'] = history[i][1]
            dispatcher_data.append(dispatcher_item)
# Write the dispatcher data to a new file
with open("dispatchertest.jsonl", "w") as dispatcher_file:
    for item in dispatcher_data:
        dispatcher_file.write(json.dumps(item) + "\n")

print("Dispatcher data saved to 'dispatchertest.jsonl'.")
del dispatcher_data
user_data = []
for item in reformatted_testdata:
    history = item['history']
    for i in range(len(history)):
        if i == 0 : continue
        if history[i][0] == 'User':
            user_item = dict(item)
            user_item['history'] = history[:i+1]
            user_item['instruction'] = history[i-1][1]
            user_item['output'] = history[i][1]
            user_data.append(user_item)
            
# Write the user data to a new file
with open("usertest.jsonl", "w") as user_file:
    for item in user_data:
        user_file.write(json.dumps(item) + "\n")

print("User data saved to 'usertest.jsonl'.")