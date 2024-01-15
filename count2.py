import json

file_name = "results_gpt35.jsonl"

# Initialize counters
count_i_r = 0
count_i_o = 0
count_overall_i = 0

with open(file_name, "r") as file:
    data = [json.loads(line) for line in file]

# Iterate over each data item
for item in data:
    i_label = item.get('i', 0)
    r_label = item.get('r', 0)
    o_label = item.get('o', 0)

    # Count for "r" is 1 when "i" is -1 / "i" is -1
    if i_label == -1 and r_label == 1:
        count_i_r += 1

    # Count for "o" is 1 when "i" is -1 / "i" is -1
    if i_label == -1 and o_label == 1:
        count_i_o += 1

    # Count overall "i" is -1
    if i_label == -1:
        count_overall_i += 1

# Calculate overall ratios
ratio_i_r = count_i_r / count_overall_i if count_overall_i > 0 else 0
ratio_i_o = count_i_o / count_overall_i if count_overall_i > 0 else 0

# Print results
print(f"Overall ratio for 'r' is 1 when 'i' is -1: {ratio_i_r:.2%}")
print(f"Overall ratio for 'o' is 1 when 'i' is -1: {ratio_i_o:.2%}")
