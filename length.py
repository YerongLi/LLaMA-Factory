import pandas as pd
import os
from sklearn.metrics import f1_score, classification_report
from scipy.stats import ttest_rel

# Specify the directory where the CSV files are located
directory_path = 'out'

# Specify the file to append F1 scores
output_file = 'f1_scores.txt'

# Specify upper and lower bounds for neutral
lower_bound = 49
upper_bound = 51

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
            ratio_ranges = [(0, 1/3), (1/3, 2/3), (2/3, 1)]

            # Iterate over ratio ranges
            for i, (lower, upper) in enumerate(ratio_ranges):
                # Filter DataFrame based on the ratio range
                df_filtered = df[(df['ratio'] >= lower) & (df['ratio'] < upper)]
                
                # Calculate ratios for o_tone_mapped
                n_o, neu_o, pos_o = calculate_ratio_for_tone(df_filtered, 'o_tone_mapped')
                
                # Calculate ratios for r_tone_mapped
                n_r, neu_r, pos_r = calculate_ratio_for_tone(df_filtered, 'r_tone_mapped')
                
                # Print the results
                print(f"For ratio range {i + 1}:")
                print(f"o_tone_mapped ratios: -1:{n_o:.2%}, 0:{neu_o:.2%}, 1:{pos_o:.2%}")
                print(f"r_tone_mapped ratios: -1:{n_r:.2%}, 0:{neu_r:.2%}, 1:{pos_r:.2%}\n")