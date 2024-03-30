import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools

# Parse command line arguments
parser = argparse.ArgumentParser(description='Your program description')
parser.add_argument('filename', type=argparse.FileType('r'))
args = parser.parse_args()
file_name = args.filename.name

# Read data from "summary.jsonl"
event_id_key_dict = {}
with open("summary_w_key.jsonl", "r") as jsonl_file:
    for line in jsonl_file:
        json_obj = json.loads(line)
        event_id = json_obj.get("event_id")
        key_value = json_obj['his_len']

        if event_id:
            event_id_key_dict[event_id] = key_value

# Read data from the provided file
with open(file_name, "r") as file:
    data = [json.loads(line) for line in file]

# Extract data for plotting from the provided file
zero_ratio = [len(line['history']) / event_id_key_dict[line['event_id']] for line in data if line['o'] == -1]
r_ratio = [len(line['history']) / event_id_key_dict[line['event_id']] for line in data if line['r'] == -1]

# Read data from "answer_gpt35.jsonl"
with open("gpt35.jsonl", "r") as jsonl_file:
    answer_gpt35_data = [json.loads(line) for line in jsonl_file]

# Extract data for plotting from "answer_gpt35.jsonl"
r_ratio_gpt35 = [len(line['history']) / event_id_key_dict[line['event_id']] for line in answer_gpt35_data if line['r'] == -1]

# Read data from "usergan.jsonl"
with open("user4.jsonl", "r") as jsonl_file:
    usergan_data = [json.loads(line) for line in jsonl_file]

# Extract data for plotting from "usergan.jsonl"
r_ratio_usergan = [len(line['history']) / event_id_key_dict[line['event_id']] for line in usergan_data if line['r'] == -1]

# Combine the data into a DataFrame, ensuring that GPT3.5 is the last entry
df = pd.DataFrame({
    'Ratio': zero_ratio + r_ratio + r_ratio_usergan + r_ratio_gpt35,
    'Victim': ['Human'] * len(zero_ratio) + ['VicSim'] * len(r_ratio_usergan) + ['VicSim w/o GAN'] * len(r_ratio) + ['GPT3.5'] * len(r_ratio_gpt35)
})

# Rearrange the DataFrame and group the ratios into the desired intervals
def adjust_ratio_group(x):
    if x <= 0.2:
        return 0
    elif x <= 0.4:
        return 1
    elif x <= 0.6:
        return 2
    elif x <= 0.8:
        return 3
    else:
        return 4

df['Ratio Group'] = df['Ratio'].apply(adjust_ratio_group)

# Use Seaborn's barplot function with hatches
hatches = itertools.cycle(['/', '\\', 'o', '*'])
fig, ax = plt.subplots(figsize=(8, 6))

sns.barplot(data=df, x="Ratio Group", y="Ratio", hue="Victim", palette={'Human': 'lightblue', 'VicSim': 'grey', 'VicSim w/o GAN': 'lightgreen', 'GPT3.5': 'salmon'}, ax=ax)

# Add hatches to the bars
for i, patch in enumerate(ax.patches):
    hatch = next(hatches)
    patch.set_hatch(hatch)

# Customize legend to show hatches for the 'Human' category
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:], loc='best')  # Exclude 'Human' from the legend

# Set x-axis and y-axis labels
plt.xlabel('Ratio Group')
plt.ylabel('Ratio')
plt.xlabel('Stages of Dialogue')
plt.ylabel('Number of Negative Expressions')
plt.savefig('distribution.png')
