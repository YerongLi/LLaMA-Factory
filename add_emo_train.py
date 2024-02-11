import json
import sys

def main(input_filename):
    # Read input JSONL file
    with open(f"data/{input_filename}.jsonl" "r") as file:
        police_data = [json.loads(line) for line in file]

    # Read results-cmp.jsonl (assuming it's the same format as police_data)
    with open("results-cmp.jsonl", "r") as results_file:
        results_data = [json.loads(line) for line in results_file]

    # Preprocess results_data into a dictionary
    preprocessed_results = {f"{data['instruction']}=={data['output']}": data for data in results_data}

    # Create a dictionary to store updated records
    updated_records = {}
    skipped_count = 0

    # Match records based on 'instruction' and 'output'
    for record in police_data:
        instruction = record.get("instruction")
        output = record.get("output")
        key = f"{instruction}=={output}"
        matching_result = preprocessed_results.get(key)
        if matching_result:
            # Update the 'response' value in the police record
            record["orignalo"] = record["output"]
            record["output"] = matching_result["response"].strip("\n")
            updated_records[key] = record
        else:
            skipped_count += 1

    # Generate output filename
    output_filename = f"data/emotional-{input_filename}.jsonl"

    # Write updated records to the output file
    with open(output_filename, "w") as output_file:
        for record in updated_records.values():
            json.dump(record, output_file)
            output_file.write("\n")

    print(f"Updated records written to {output_filename}. Skipped {skipped_count} records.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_emo_train.py <input_filename>")
        sys.exit(1)
    input_filename = sys.argv[1]
    main(input_filename)
