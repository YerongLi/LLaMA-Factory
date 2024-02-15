import json

# Initialize an empty dictionary to store event summaries
event_summaries = {}

# Open the "summary.jsonl" file
with open("summary.jsonl", "r") as file:
    # Read each line (which is a JSON object)
    for line in file:
        # Load the JSON object
        data = json.loads(line)
        
        # Extract event ID and summary
        event_id = data.get("event_id")
        summary = data.get("summary")
        
        # Add the event summary to the dictionary
        if event_id and summary:
            event_summaries[event_id] = summary

# Print the constructed dictionary
print(event_summaries)
# Read the user data from the 'user.jsonl' file
with open("user.jsonl", "r") as user_file:
    user_data = [json.loads(line) for line in user_file]

    # Update the summaries in user_data using event_summaries
    for item in user_data:
        event_id = item.get("event_id")
        if event_id in event_summaries:
            item["summary"] = event_summaries[event_id]

# Write the updated user data back to the original file
with open("user.jsonl", "w") as updated_user_file:
    for item in user_data:
        updated_user_file.write(json.dumps(item) + "\n")

print("Updated user data saved to 'user.jsonl'.")