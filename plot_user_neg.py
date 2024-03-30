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
# sns.set_context(rc = {'patch.linewidth': 1})

# sns.set(style="whitegrid", color_codes=True)
df = pd.DataFrame({
    'Ratio': zero_ratio + r_ratio + r_ratio_usergan + r_ratio_gpt35,
    'Model': ['Human'] * len(zero_ratio) + ['VicSim'] * len(r_ratio_usergan) + ['VicSim w/o GAN'] * len(r_ratio) + ['GPT3.5'] * len(r_ratio_gpt35),
    'total': [1] * len(zero_ratio) + [1] * len(r_ratio_usergan) + [1] * len(r_ratio) + [1] * len(r_ratio_gpt35)
})
# print(df)
def adjust_ratio(x):
    x += 0.5
    if x > 1:
        x -= 1
    return x
def map_to_ratio_group(ratio):
    if ratio <= 0.2:
        return 0
    elif ratio <= 0.4:
        return 1
    elif ratio <= 0.6:
        return 2
    elif ratio <= 0.8:
        return 3
    else:
        return 4
df['Ratio'] = df.apply(lambda row: adjust_ratio(row['Ratio']) if row['Model'] != 'GPT3.5' else row['Ratio'], axis=1)
# Create DataFrame with mapped ratio groups and renamed column

# Map 'Ratio' to 'Ratio_Group' and rename column
df['Ratio_Group'] = df['Ratio'].apply(map_to_ratio_group)
df.drop(columns=['Ratio'], inplace=True)  # Drop the original 'Ratio' column
# sns.set(style="whitegrid")
# ax = sns.histplot(data=df, x="Ratio", hue="Victim", palette={'Human': 'lightblue', 'VicSim': 'grey', 'VicSim w/o GAN': 'lightgreen', 'GPT3.5': 'salmon'}, multiple="dodge", bins=5, element="bars", shrink=0.6)
fig, ax = plt.subplots()
sns.barplot(data=df, x="Ratio_Group", y='total', hue='Model', estimator=sum, palette={'Human': 'lightblue', 'VicSim': 'grey', 'VicSim w/o GAN': 'lightgreen', 'GPT3.5': 'salmon'},linewidth=2)
hatches = ['/', '\\', 'o', '*']
# hatches = itertools.cycle(['/', '//', '+', '-', 'x', '\\', '*', 'o', 'O', '.'])
# Customize x-axis and y-axis
plt.gca().spines['bottom'].set_color('black')  # Darken x-axis
plt.gca().spines['left'].set_color('black')    # Darken y-axis

# Darken tick marks and lines on the x-axis and y-axis
plt.tick_params(axis='x', colors='black', which='both')
plt.tick_params(axis='y', colors='black', which='both')

# for hues, hatch in zip(ax.containers, hatches):
#     # set a different hatch for each time
#     for hue in hues:
#         hue.set_hatch(hatch)
for container, hatch, handle in zip(ax.containers, hatches, ax.get_legend().legend_handles, ):
    
    # update the hatching in the legend handle
    handle.set_hatch(hatch)
    
    # iterate through each rectangle in the container
    for rectangle in container:

        # set the rectangle hatch
        rectangle.set_hatch(hatch)

plt.xticks(ticks=[i for i in range(5)], labels=[f'{(i +1)* 20}%' for i in range(5)])
# plt.legend(title='Model')

plt.xlabel('Stages of Dialogue')
plt.ylabel('Number of Negative Expressions')
# plt.title('Distribution of negative responses from human, VicSim, Llama, and GPT-3.5 responses at different stages of dialogues')

plt.savefig('distribution.png')