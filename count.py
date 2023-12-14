import pandas as pd
import os
from sklearn.metrics import f1_score

# Specify the directory where the CSV files are located
directory_path = 'out'

# Specify the file to append F1 scores
output_file = 'f1_scores.txt'

# Specify upper and lower bounds for neutral
lower_bound = 30
upper_bound = 70

# Open the file in append mode
with open(output_file, 'a') as f:
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

            # Count occurrences where 'i_tone', 'r_tone', and 'o_tone' are between 30 and 70
            count_neutral = {
                'i_tone': ((df['i_tone'] >= lower_bound) & (df['i_tone'] <= upper_bound)).sum(),
                'r_tone': ((df['r_tone'] >= lower_bound) & (df['r_tone'] <= upper_bound)).sum(),
                'o_tone': ((df['o_tone'] >= lower_bound) & (df['o_tone'] <= upper_bound)).sum()
            }

            # Print the original counts for each file
            print(f"File: {filename}")
            print(f"Count of 'i_tone' values above 70: {count_above_70['i_tone']}")
            print(f"Count of 'r_tone' values above 70: {count_above_70['r_tone']}")
            print(f"Count of 'o_tone' values above 70: {count_above_70['o_tone']}")

            print(f"Count of 'i_tone' values below 30: {count_below_30['i_tone']}")
            print(f"Count of 'r_tone' values below 30: {count_below_30['r_tone']}")
            print(f"Count of 'o_tone' values below 30: {count_below_30['o_tone']}")

            print(f"Count of 'i_tone' values between {lower_bound} and {upper_bound} (neutral): {count_neutral['i_tone']}")
            print(f"Count of 'r_tone' values between {lower_bound} and {upper_bound} (neutral): {count_neutral['r_tone']}")
            print(f"Count of 'o_tone' values between {lower_bound} and {upper_bound} (neutral): {count_neutral['o_tone']}")

            # Define conditions for positive, negative, and neutral
            positive_condition_o_tone = df['o_tone'] > 70
            positive_condition_r_tone = df['r_tone'] > 70
            negative_condition_o_tone = df['o_tone'] < 30
            negative_condition_r_tone = df['r_tone'] < 30

            # Count occurrences where 'o_tone' is negative or neutral while 'r_tone' is positive
            count_negative_or_neutral_o_tone_positive_r_tone = (negative_condition_o_tone | (df['o_tone'] == 0)) & positive_condition_r_tone
            count_negative_or_neutral_o_tone_positive_r_tone = count_negative_or_neutral_o_tone_positive_r_tone.sum()

            # Count occurrences where 'r_tone' is negative or neutral while 'o_tone' is positive
            count_negative_or_neutral_r_tone_positive_o_tone = (negative_condition_r_tone | (df['r_tone'] == 0)) & positive_condition_o_tone
            count_negative_or_neutral_r_tone_positive_o_tone = count_negative_or_neutral_r_tone_positive_o_tone.sum()

            # Print the additional counts
            print(f"Count where 'o_tone' is negative or neutral while 'r_tone' is positive: {count_negative_or_neutral_o_tone_positive_r_tone}")
            print(f"Count where 'r_tone' is negative or neutral while 'o_tone' is positive: {count_negative_or_neutral_r_tone_positive_o_tone}")

            # Define conditions for positive, negative, and neutral
            positive_condition = df['o_tone'] > 70
            negative_condition_i_tone = (df['i_tone'] < 0) & (df['o_tone'] > 0)
            negative_condition_r_tone = (df['i_tone'] < 0) & (df['r_tone'] > 0)

            # Assign true labels based on conditions
            true_labels = [1 if pos else (-1 if neg_i else 0) for pos, neg_i in zip(positive_condition, negative_condition_i_tone)]

            # Assign predicted labels based on 'r_tone' predictions
            predicted_labels = [1 if val > 70 else (-1 if val < 30 else 0) for val in df['r_tone']]

            # Calculate F1 score for the current file
            f1 = f1_score(true_labels, predicted_labels, average='weighted') * 100

            # Print the F1 score for the current file
            print(f"F1 Score for {filename}: {f1:.2f}")

            # Append the results to the output file
            f.write(f"File: {filename}, F1 Score: {f1:.2f}\n")

            # Count occurrences where i_tone is negative while o_tone is positive
            count_negative_i_tone_positive_o_tone = (negative_condition_i_tone).sum()

            # Count occurrences where i_tone is negative while r_tone is positive
            count_negative_i_tone_positive_r_tone = (negative_condition_r_tone).sum()

            # Print the additional counts
            print(f"Count where 'i_tone' is negative while 'o_tone' is positive: {count_negative_i_tone_positive_o_tone}")
            print(f"Count where 'i_tone' is negative while 'r_tone' is positive: {count_negative_i_tone_positive_r_tone}")

            print("\n")
