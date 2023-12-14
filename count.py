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

            # Replace 0 values in 'o_tone' with NaN
            # df['o_tone'].replace(0, pd.NA, inplace=True)

            # Count occurrences where 'i_tone', 'r_tone', and 'o_tone' are above upper_bound
            count_above_upper_bound = {
                'i_tone': (df['i_tone'] > upper_bound).sum(),
                'r_tone': (df['r_tone'] > upper_bound).sum(),
                'o_tone': (df['o_tone'] > upper_bound).sum()
            }

            # Count occurrences where 'i_tone', 'r_tone', and 'o_tone' are below lower_bound
            count_below_lower_bound = {
                'i_tone': (df['i_tone'] < lower_bound).sum(),
                'r_tone': (df['r_tone'] < lower_bound).sum(),
                'o_tone': (df['o_tone'] < lower_bound).sum()
            }

            # Count occurrences where 'i_tone', 'r_tone', and 'o_tone' are between lower_bound and upper_bound
            count_neutral = {
                'i_tone': ((df['i_tone'] >= lower_bound) & (df['i_tone'] <= upper_bound)).sum(),
                'r_tone': ((df['r_tone'] >= lower_bound) & (df['r_tone'] <= upper_bound)).sum(),
                'o_tone': ((df['o_tone'] >= lower_bound) & (df['o_tone'] <= upper_bound)).sum()
            }

            # Print the original counts for each file
            print(f"File: {filename}")
            print(f"Count of 'i_tone' values above {upper_bound}: {count_above_upper_bound['i_tone']}")
            print(f"Count of 'r_tone' values above {upper_bound}: {count_above_upper_bound['r_tone']}")
            print(f"Count of 'o_tone' values above {upper_bound}: {count_above_upper_bound['o_tone']}")

            print(f"Count of 'i_tone' values below {lower_bound}: {count_below_lower_bound['i_tone']}")
            print(f"Count of 'r_tone' values below {lower_bound}: {count_below_lower_bound['r_tone']}")
            print(f"Count of 'o_tone' values below {lower_bound}: {count_below_lower_bound['o_tone']}")

            print(f"Count of 'i_tone' values between {lower_bound} and {upper_bound} (neutral): {count_neutral['i_tone']}")
            print(f"Count of 'r_tone' values between {lower_bound} and {upper_bound} (neutral): {count_neutral['r_tone']}")
            print(f"Count of 'o_tone' values between {lower_bound} and {upper_bound} (neutral): {count_neutral['o_tone']}")

            # Define conditions for positive, negative, and neutral
            positive_condition_o_tone = df['o_tone'] > upper_bound
            positive_condition_r_tone = df['r_tone'] > upper_bound
            negative_condition_o_tone = df['o_tone'] < lower_bound
            negative_condition_r_tone = df['r_tone'] < lower_bound

            # Count occurrences where 'o_tone' is negative or neutral while 'r_tone' is positive
            count_negative_or_neutral_o_tone_positive_r_tone = (negative_condition_o_tone | pd.isna(df['o_tone'])) & positive_condition_r_tone
            count_negative_or_neutral_o_tone_positive_r_tone = count_negative_or_neutral_o_tone_positive_r_tone.sum()

            # Count occurrences where 'r_tone' is negative or neutral while 'o_tone' is positive
            count_negative_or_neutral_r_tone_positive_o_tone = (negative_condition_r_tone | pd.isna(df['r_tone'])) & positive_condition_o_tone
            count_negative_or_neutral_r_tone_positive_o_tone = count_negative_or_neutral_r_tone_positive_o_tone.sum()

            # Print the additional counts
            print(f"Count where 'o_tone' is negative or neutral while 'r_tone' is positive: {count_negative_or_neutral_o_tone_positive_r_tone}")
            print(f"Count where 'r_tone' is negative or neutral while 'o_tone' is positive: {count_negative_or_neutral_r_tone_positive_o_tone}")

            # Define conditions for positive, negative, and neutral
            positive_condition = df['o_tone'] > upper_bound
            negative_condition_i_tone = (df['i_tone'] < 0) & (df['o_tone'] > 0)
            negative_condition_r_tone = (df['i_tone'] < 0) & (df['r_tone'] > 0)

            # Assign true labels based on conditions
            true_labels = [1 if pos else (-1 if neg_i else 0) for pos, neg_i in zip(positive_condition, negative_condition_i_tone)]

            # Assign predicted labels based on 'r_tone' predictions
            predicted_labels = [1 if val > upper_bound else (-1 if val < lower_bound else 0) for val in df['r_tone']]

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
