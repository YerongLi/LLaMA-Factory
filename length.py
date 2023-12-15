import pandas as pd
import os
from sklearn.metrics import f1_score, classification_report
from scipy.stats import ttest_rel

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
            length = df['his_len']
            df['current_length'] = df['history'].apply(lambda x: len(eval(x)))
            df['ratio'] = df['current_length'] / df['his_len']
            # print(df['ratio'])
            # Filter df for each ratio range
            df_less_than_third = df[df['ratio'] < 1/3]
            df_between_third_and_two_thirds = df[(df['ratio'] >= 1/3) & (df['ratio'] < 2/3)]
            df_greater_than_two_thirds = df[df['ratio'] >= 2/3]

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

            # Calculate ratios for o_tone_mapped in each ratio range
            ratio_negative_o_less_than_third, ratio_neutral_o_less_than_third, ratio_positive_o_less_than_third = calculate_ratio_for_tone(df_less_than_third, 'o_tone_mapped')
            ratio_negative_o_between_third_and_two_thirds, ratio_neutral_o_between_third_and_two_thirds, ratio_positive_o_between_third_and_two_thirds = calculate_ratio_for_tone(df_between_third_and_two_thirds, 'o_tone_mapped')
            ratio_negative_o_greater_than_two_thirds, ratio_neutral_o_greater_than_two_thirds, ratio_positive_o_greater_than_two_thirds = calculate_ratio_for_tone(df_greater_than_two_thirds, 'o_tone_mapped')

            # Calculate ratios for r_tone_mapped in each ratio range
            ratio_negative_r_less_than_third, ratio_neutral_r_less_than_third, ratio_positive_r_less_than_third = calculate_ratio_for_tone(df_less_than_third, 'r_tone_mapped')
            ratio_negative_r_between_third_and_two_thirds, ratio_neutral_r_between_third_and_two_thirds, ratio_positive_r_between_third_and_two_thirds = calculate_ratio_for_tone(df_between_third_and_two_thirds, 'r_tone_mapped')
            ratio_negative_r_greater_than_two_thirds, ratio_neutral_r_greater_than_two_thirds, ratio_positive_r_greater_than_two_thirds = calculate_ratio_for_tone(df_greater_than_two_thirds, 'r_tone_mapped')

            # Print the results
            print("For ratio < 1/3:")
            print(f"o_tone_mapped ratios: -1:{ratio_negative_o_less_than_third:.2%}, 0:{ratio_neutral_o_less_than_third:.2%}, 1:{ratio_positive_o_less_than_third:.2%}")
            print(f"r_tone_mapped ratios: -1:{ratio_negative_r_less_than_third:.2%}, 0:{ratio_neutral_r_less_than_third:.2%}, 1:{ratio_positive_r_less_than_third:.2%}\n")

            print("For 1/3 <= ratio < 2/3:")
            print(f"o_tone_mapped ratios: -1:{ratio_negative_o_between_third_and_two_thirds:.2%}, 0:{ratio_neutral_o_between_third_and_two_thirds:.2%}, 1:{ratio_positive_o_between_third_and_two_thirds:.2%}")
            print(f"r_tone_mapped ratios: -1:{ratio_negative_r_between_third_and_two_thirds:.2%}, 0:{ratio_neutral_r_between_third_and_two_thirds:.2%}, 1:{ratio_positive_r_between_third_and_two_thirds:.2%}\n")

            print("For ratio >= 2/3:")
            print(f"o_tone_mapped ratios: -1:{ratio_negative_o_greater_than_two_thirds:.2%}, 0:{ratio_neutral_o_greater_than_two_thirds:.2%}, 1:{ratio_positive_o_greater_than_two_thirds:.2%}")
            print(f"r_tone_mapped ratios: -1:{ratio_negative_r_greater_than_two_thirds:.2%}, 0:{ratio_neutral_r_greater_than_two_thirds:.2%}, 1:{ratio_positive_r_greater_than_two_thirds:.2%}")
