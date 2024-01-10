import pandas as pd
import os
from sklearn.metrics import f1_score, classification_report
from scipy.stats import ttest_rel

# Specify the directory where the CSV files are located
directory_path = 'out'

# Specify the file to append F1 scores
output_file = 'f1_scores.txt'

# Specify upper and lower bounds for neutral
lower_bound = 40
upper_bound = 60

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

def scale(value):
  
    left, right = 0, 100  # Adjust the range based on your requirements
    while left < right:
        mid = (left + right) // 2
        mid_value = 10 ** mid * value
        if mid_value > 1:
            right = mid
        else:
            left = mid + 1
    k = left
    # Find the first non-zero digit in the original value
    value_str = str(value)
    for digit in value_str:
        if digit != '0' and digit != '.':
            first_digit = int(digit)
            break
    # print(k, first_digit, value)
    remainder = first_digit % 4

    if remainder == 0:
        scale_factor = max(0, (k - 2))
        scaled_value = value * (10 ** scale_factor)
        return scaled_value
    elif remainder == 2:
        return value
    else:

        scale_factor = max(0, (k - 3))
        scaled_value = value * (10 ** scale_factor)

        return scaled_value

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
event_types = ['SuspiciousActivity', 'AccidentTrafficParking', 'DrugsAlcohol', 'EmergencyMessage', 'FacilitiesMaintenance', 'HarassmentAbuse', 'MentalHealth', 'NoiseDisturbance', 'TheftLostItem']

with open(output_file, 'a') as f:
    # Loop through each CSV file in the directory
    all_files = [
    'group_LIWC-22 Results - all - LIWC Analysis.csv',
]

    for filename in all_files:
    # for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            # continue
            file_path = os.path.join(directory_path, filename)

            # Read the CSV file into a pandas DataFrame
            whole_df = pd.read_csv(file_path)
            # event_types = whole_df['type'].unique()
            # event_types = ['SuspiciousActivity', 'DrugsAlcohol', 'EmergencyMessage', 'FacilitiesMaintenance', 'HarassmentAbuse',
            # 'MentalHealth', 'TheftLostItem', 'SafeRide&SafeWalk']
            # event_types = ['SuspiciousActivity', 'AccidentTrafficParking', 'DrugsAlcohol', 'EmergencyMessage', 'FacilitiesMaintenance', 'HarassmentAbuse', 'MentalHealth', 'NoiseDisturbance', 'TheftLostItem']

            # print(event_types)
            p_values_tone_mapped = {}
            p_values_tone = {}
            errors_tone_mapped = {}
            errors_tone = {}

    
 
            # Map 'o_tone', 'r_tone', and 'i_tone' values to 0, 1, -1 based on conditions
            whole_df['o_tone_mapped'] = whole_df['o_tone'].apply(lambda x: 0 if (pd.isna(x) or (x >= lower_bound and x <= upper_bound)) else (1 if x > upper_bound else -1))
            whole_df['r_tone_mapped'] = whole_df['r_tone'].apply(lambda x: 0 if (pd.isna(x) or (x >= lower_bound and x <= upper_bound)) else (1 if x > upper_bound else -1))
            whole_df['i_tone_mapped'] = whole_df['i_tone'].apply(lambda x: 0 if (pd.isna(x) or (x >= lower_bound and x <= upper_bound)) else (1 if x > upper_bound else -1))
            whole_df['o_tone'].fillna(50, inplace=True)
            whole_df['r_tone'].fillna(50, inplace=True)

            if not (whole_df['hour'].ge(0) & whole_df['hour'].le(23)).all():
                print(f"Error: Values in 'hour' column are not within the range 0 to 23 for file: {filename}")
                continue

            # Calculate distribution of whole_df['o_tone_mapped'] == 1 and whole_df['r_tone_mapped'] == 1 over whole_df['hour']
            distribution_o_tone_mapped = whole_df[whole_df['o_tone_mapped'] == 1]['hour'].value_counts(normalize=True).sort_index()
            distribution_r_tone_mapped = whole_df[whole_df['r_tone_mapped'] == 1]['hour'].value_counts(normalize=True).sort_index()

            print(f"\nDistribution of 'o_tone_mapped' == 1 over 'hour' for file: {filename}")
            print(distribution_o_tone_mapped)

            print(f"\nDistribution of 'r_tone_mapped' == 1 over 'hour' for file: {filename}")
            print(distribution_r_tone_mapped)

   