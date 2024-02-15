import json

# Initialize an empty list to store the reformatted data
reformatted_data = []

# Open the original file
with open("fill_1160_70b_postprocessed.jsonl", "r") as file:
    # Read each line (which is a JSON object)
    for line in file:
        # Load the JSON object
        data = json.loads(line)
        
        # Convert the list of dictionaries to a list of lists
        reformatted_list = [[key, value] for item in data['parsed_response'] for key, value in item.items()]
        
        # Assign the reformatted list back to data['parsed_response']
        data['parsed_response'] = reformatted_list
        
        # Append the updated data to the reformatted_data list
        reformatted_data.append(data)

# Write the reformatted data to a new file
with open("fill-police-complete.jsonl", "w") as new_file:
    for item in reformatted_data:
        new_file.write(json.dumps(item) + "\n")

print("Data saved to 'fill-police-complete.jsonl'.")
