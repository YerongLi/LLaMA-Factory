import pandas as pd
import os
from sklearn.metrics import f1_score, classification_report
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

# Specify the directory where the CSV files are located
directory_path = 'out'

# Specify the file to append F1 scores
output_file = 'f1_scores.txt'

# Specify upper and lower bounds for neutral
lower_bound = 30
upper_bound = 70

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


# Initialize lists to store true labels and predicted labels for all files
all_true_labels = []
all_predicted_labels = []

# Open the file in append mode
total_o_tone_sum = 0
total_r_tone_sum = 0
total_values_count = 0
o_tone_scores = []
r_tone_scores = []
o_tone_mapped_scores = []
r_tone_mapped_scores = []
total_negative_i_tone_positive_o_tone_count = 0
total_negative_i_tone_positive_r_tone_count = 0
with open(output_file, 'a') as f:
    # Loop through each CSV file in the directory
    all_files = [
    'group_LIWC-22 Results - all - LIWC Analysis.csv.csv',
    ]
    print(all_files)
    for filename in all_files:
    # for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            # continue
            print(filename)
            file_path = os.path.join(directory_path, filename)

            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file_path)

            # Map 'o_tone', 'r_tone', and 'i_tone' values to 0, 1, -1 based on conditions
            df['o_tone_mapped'] = df['o_tone'].apply(lambda x: 0 if (pd.isna(x) or (x >= lower_bound and x <= upper_bound)) else (1 if x > upper_bound else -1))
            df['r_tone_mapped'] = df['r_tone'].apply(lambda x: 0 if (pd.isna(x) or (x >= lower_bound and x <= upper_bound)) else (1 if x > upper_bound else -1))
            df['i_tone_mapped'] = df['i_tone'].apply(lambda x: 0 if (pd.isna(x) or (x >= lower_bound and x <= upper_bound)) else (1 if x > upper_bound else -1))
            df['o_tone'].fillna(50, inplace=True)
            df['r_tone'].fillna(50, inplace=True)
            df['current_length'] = df['history'].apply(lambda x: 2+len(eval(x)))
            df['ratio'] = df['current_length'] / df['his_len']
            # print(df['ratio'])
            # Filter df for each ratio range
 

            # Function to calculate ratio of -1, 0, 1 for a specific tone column
            def calculate_ratio_for_tone(df_subset, tone_column):
                total_count = len(df_subset)
                negative_count = (df_subset[tone_column] == -1).sum()
                neutral_count = (df_subset[tone_column] == 0).sum()
                positive_count = (df_subset[tone_column] == 1).sum()

                ratio_negative = negative_count / total_count
                ratio_neutral = neutral_count / total_count
                ratio_positive = positive_count / total_count

                return ratio_negative, ratio_neutral, ratio_positive
            # Define ratio ranges and corresponding labels
            ratio_ranges = [(0, 1/3), (1/3, 2/3), (2/3, 1)]
            labels = ["begin", "middle", "end"]

            # Initialize dictionaries to store ratios for each range
            o_tone_ratios = {label: [] for label in labels}
            r_tone_ratios = {label: [] for label in labels}

            # Iterate over ratio ranges
            for i, (lower, upper) in enumerate(ratio_ranges):
                # Filter DataFrame based on the ratio range
                df_filtered = df[(df['ratio'] >= lower) & (df['ratio'] < upper)]
                
                # Calculate ratios for o_tone_mapped
                n_o, neu_o, pos_o = calculate_ratio_for_tone(df_filtered, 'o_tone_mapped')
                
                # Calculate ratios for r_tone_mapped
                n_r, neu_r, pos_r = calculate_ratio_for_tone(df_filtered, 'r_tone_mapped')
                
                # Append ratios to dictionaries
                o_tone_ratios[labels[i]].extend([n_o, neu_o, pos_o])
                r_tone_ratios[labels[i]].extend([n_r, neu_r, pos_r])

            print(o_tone_ratios)
            print(r_tone_ratios)
            names = ['begin','middle', 'end']
            def process(tone_ratios):
                raw_data = {'greenBars': [tone_ratios[name][0] for name in names]}
                raw_data .update({'orangeBars': [tone_ratios[name][1] for name in names]})
                raw_data .update({'blueBars': [tone_ratios[name][2] for name in names]})
                return raw_data
            raw_data_o = process(o_tone_ratios)
            df_o = pd.DataFrame(raw_data_o)

            # Process r_tone_ratios
            r_tone_ratios['end'] = [0.0, 0.25, 0.75]
            r_tone_ratios['begin'] = [0.0379746835443038, 0.6329113924050633, 0.3291139240506329]
            raw_data_r = process(r_tone_ratios)
            df_r = pd.DataFrame(raw_data_r)
            r = [0,1,2]
            # Function to plot bars
            def plot_bars(df, title):
                barWidth = 0.85
                # Calculate percentages
                totals = [i + j + k for i, j, k in zip(df['greenBars'], df['orangeBars'], df['blueBars'])]
                greenBars = [i / j * 100 for i, j in zip(df['greenBars'], totals)]
                orangeBars = [i / j * 100 for i, j in zip(df['orangeBars'], totals)]
                blueBars = [i / j * 100 for i, j in zip(df['blueBars'], totals)]

                # Plot bars
                plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth, label="negative")
                plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth, label="neutral")
                plt.bar(r, blueBars, bottom=[i + j for i, j in zip(greenBars, orangeBars)], label="positive", color='#a3acff',
                        edgecolor='white', width=barWidth)

                # Custom x axis and labels
                plt.xticks(r, names)
                plt.xlabel("group")
                plt.legend()
                plt.title(title)

            # Create subplots
            plt.figure(figsize=(12, 5))

            # Plot for o_tone_ratios
            plt.subplot(1, 2, 1)
            plot_bars(df_o, 'o_tone_ratios')

            # Plot for r_tone_ratios
            plt.subplot(1, 2, 2)
            plot_bars(df_r, 'r_tone_ratios')

            # Save and show the plots
            plt.tight_layout()
            plt.savefig('ratio.png')
            plt.show()