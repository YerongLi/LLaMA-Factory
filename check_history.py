import json
import sys

def check_history_matches(input_filename):
    # Read input JSONL file
    with open(f"data/{input_filename}.jsonl", "r") as file:
        police_data = [json.loads(line) for line in file]

    for entry in police_data:
        # Check if the last history entry matches the output
        if entry['history'][-1][1] == entry['output']:
            print(f"History matches output for entry {entry['id']}")
        else:
            print(f"Mismatch: History does not match output for entry {entry['id']}")

        # Check if the second-to-last history entry matches the instruction
        if entry['history'][-2][1] == entry['instruction']:
            print(f"History matches instruction for entry {entry['id']}")
        else:
            print(f"Mismatch: History does not match instruction for entry {entry['id']}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_history.py <input_filename>")
        sys.exit(1)
    input_filename = sys.argv[1]
    check_history_matches(input_filename)
