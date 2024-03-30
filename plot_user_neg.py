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

def adjust_ratio(x):
    x += 0.5
    if x > 1:
        x -= 1
    return x

df['Ratio'] = df.apply(lambda row: adjust_ratio(row['Ratio']) if row['Victim'] != 'GPT3.5' else row['Ratio'], axis=1)

# Rearrange the DataFrame
df = df.pivot_table(index='Victim', columns='Ratio', aggfunc='size', fill_value=0)

# Reset index for plotting
df = df.reset_index()

# sns.set(style="whitegrid")
ax = sns.barplot(data=df, x="Victim", y=0, color='lightblue')
ax = sns.barplot(data=df, x="Victim", y=1, color='grey')
ax = sns.barplot(data=df, x="Victim", y=2, color='lightgreen')
ax = sns.barplot(data=df, x="Victim", y=3, color='salmon')

# Customize x-axis and y-axis
plt.gca().spines['bottom'].set_color('black')  # Darken x-axis
plt.gca().spines['left'].set_color('black')    # Darken y-axis

# Darken tick marks and lines on the x-axis and y-axis
plt.tick_params(axis='x', colors='black', which='both')
plt.tick_params(axis='y', colors='black', which='both')

# Add hatches
hatches = itertools.cycle(['/', '\\', 'o', '*'])
for i, patch in enumerate(ax.patches):
    if i % 4 == 3:
        hatch = next(hatches)
    patch.set_hatch(hatch)

# Set legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[0:1], ['Human'])

plt.xlabel('Stages of Dialogue')
plt.ylabel('Number of Negative Expressions')

plt.savefig('distribution.png')