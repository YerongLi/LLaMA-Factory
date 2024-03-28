import json

# Input and output file paths
input_file_path = "answer_gpt35.jsonl"
output_file_path = "gpt35.jsonl"

# Open input file for reading and output file for writing
with open(input_file_path, "r") as input_file, open(output_file_path, "w") as output_file:
    # Iterate through each line in the input file
    for line in input_file:
        # Load JSON object from the current line
        json_obj = json.loads(line)
        
        # Rename the field "gpt35_response" to "response"
        json_obj["response"] = json_obj.pop("gpt35_response")
        
        # Write the modified JSON object to the output file
        output_file.write(json.dumps(json_obj) + "\n")
