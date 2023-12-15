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
            print(length)