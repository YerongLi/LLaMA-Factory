import pandas as pd
import os
from sklearn.metrics import f1_score, confusion_matrix

# Specify the directory where the CSV files are located
directory_path = 'out'

# Loop through each CSV file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # Count occurrences where 'i_tone', 'r_tone', and 'o_tone' are above 70
        count_above_70 = {
            'i_tone': (df['i_tone'] > 70).sum(),
            'r_tone': (df['r_tone'] > 70).sum(),
            'o_tone': (df['o_tone'] > 70).sum()
        }

        # Count occurrences where 'i_tone', 'r_tone', and 'o_tone' are below 30
        count_below_30 = {
            'i_tone': (df['i_tone'] < 30).sum(),
            'r_tone': (df['r_tone'] < 30).sum(),
            'o_tone': (df['o_tone'] < 30).sum()
        }
        df = pd.read_csv(file_path)

        # Define conditions for positive, negative, and neutral
        positive_condition = df['o_tone'] > 70
        negative_condition = df['o_tone'] < 30

        # Assign true labels based on conditions
        true_labels.extend([1 if pos else (-1 if neg else 0) for pos, neg in zip(positive_condition, negative_condition)])

        # Assign predicted labels based on 'r_tone' predictions
        predicted_labels.extend([1 if val > 70 else (-1 if val < 30 else 0) for val in df['r_tone']])
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        # Print the results for each file
        print(f"File: {filename}")
        print(f"Count of 'i_tone' values above 70: {count_above_70['i_tone']}")
        print(f"Count of 'r_tone' values above 70: {count_above_70['r_tone']}")
        print(f"Count of 'o_tone' values above 70: {count_above_70['o_tone']}")

        print(f"Count of 'i_tone' values below 30: {count_below_30['i_tone']}")
        print(f"Count of 'r_tone' values below 30: {count_below_30['r_tone']}")
        print(f"Count of 'o_tone' values below 30: {count_below_30['o_tone']}")
        print(f"F1 Score: {f1}")

        print("\n")
