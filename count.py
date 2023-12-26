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
    for filename in all_files:
    # for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            # continue
            file_path = os.path.join(directory_path, filename)

            # Read the CSV file into a pandas DataFrame
            whole_df = pd.read_csv(file_path)
            event_types = whole_df['type'].unique()
            for event_type in event_types + [None]:
                if event_type is not None:
                    df = whole_df[whole_df['type'] == event_type]
                else:
                    df = whole_df

                # Map 'o_tone', 'r_tone', and 'i_tone' values to 0, 1, -1 based on conditions
                df['o_tone_mapped'] = df['o_tone'].apply(lambda x: 0 if (pd.isna(x) or (x >= lower_bound and x <= upper_bound)) else (1 if x > upper_bound else -1))
                df['r_tone_mapped'] = df['r_tone'].apply(lambda x: 0 if (pd.isna(x) or (x >= lower_bound and x <= upper_bound)) else (1 if x > upper_bound else -1))
                df['i_tone_mapped'] = df['i_tone'].apply(lambda x: 0 if (pd.isna(x) or (x >= lower_bound and x <= upper_bound)) else (1 if x > upper_bound else -1))
                df['o_tone'].fillna(50, inplace=True)
                df['r_tone'].fillna(50, inplace=True)
                avg_o_tone = df['o_tone'].mean()
                avg_r_tone = df['r_tone'].mean()
                total_o_tone_sum += df['o_tone'].sum()
                total_r_tone_sum += df['r_tone'].sum()
                total_values_count += len(df)  # Count the number of rows
                # Define conditions for positive, negative, and neutral
                positive_condition_o_tone = df['o_tone_mapped'] == 1
                positive_condition_r_tone = df['r_tone_mapped'] == 1
                negative_condition_o_tone = df['o_tone_mapped'] == -1
                negative_condition_r_tone = df['r_tone_mapped'] == -1

                negative_i_tone_positive_o_tone_condition = (df['i_tone_mapped'] == -1) & (df['o_tone_mapped'] == 1)

                # Define conditions for negative 'i_tone' while 'r_tone' is positive
                negative_i_tone_positive_r_tone_condition = (df['i_tone_mapped'] == -1) & (df['r_tone_mapped'] == 1)

                # Count occurrences for the current file
                count_negative_i_tone_positive_o_tone = negative_i_tone_positive_o_tone_condition.sum()
                count_negative_i_tone_positive_r_tone = negative_i_tone_positive_r_tone_condition.sum()

                # Print results for the current file
                print(f"Percentage where 'i_tone' is negative while 'o_tone' is positive: {(count_negative_i_tone_positive_o_tone / len(df)) * 100:.2f}")
                print(f"Percentage where 'i_tone' is negative while 'r_tone' is positive: {(count_negative_i_tone_positive_r_tone / len(df)) * 100:.2f}")
                print("\n")

                # Accumulate counts for overall percentages
                total_negative_i_tone_positive_o_tone_count += count_negative_i_tone_positive_o_tone
                total_negative_i_tone_positive_r_tone_count += count_negative_i_tone_positive_r_tone



                # Count occurrences where 'o_tone' is negative or neutral while 'r_tone' is positive
                count_negative_or_neutral_o_tone_positive_r_tone = (negative_condition_o_tone | (df['o_tone_mapped'] == 0)) & positive_condition_r_tone
                count_negative_or_neutral_o_tone_positive_r_tone = count_negative_or_neutral_o_tone_positive_r_tone.sum()

                # Count occurrences where 'r_tone' is negative or neutral while 'o_tone' is positive
                count_negative_or_neutral_r_tone_positive_o_tone = (negative_condition_r_tone | (df['r_tone_mapped'] == 0)) & positive_condition_o_tone
                count_negative_or_neutral_r_tone_positive_o_tone = count_negative_or_neutral_r_tone_positive_o_tone.sum()

                # Define conditions for positive, negative, and neutral
                positive_condition = df['o_tone_mapped'] == 1
                negative_condition_i_tone = (df['i_tone_mapped'] == -1) & (df['o_tone_mapped'] == 1)
                negative_condition_r_tone = (df['r_tone_mapped'] == -1) & (df['o_tone_mapped'] == 1)

                # Assign true labels based on conditions
                true_labels = [1 if pos else (-1 if neg_i else 0) for pos, neg_i in zip(positive_condition, negative_condition_i_tone)]

                # Assign predicted labels based on 'r_tone' predictions
                predicted_labels = [1 if val == 1 else (-1 if val == -1 else 0) for val in df['r_tone_mapped']]

                # Append true and predicted labels to the lists
                all_true_labels.extend(true_labels)
                all_predicted_labels.extend(predicted_labels)

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
                print(filename)

                print(f"Average 'o_tone': {avg_o_tone:.2f}")
                print(f"Average 'r_tone': {avg_r_tone:.2f}")
                print(f"Count where 'i_tone' is negative while 'o_tone' is positive: {count_negative_i_tone_positive_o_tone}")
                print(f"Count where 'i_tone' is negative while 'r_tone' is positive: {count_negative_i_tone_positive_r_tone}")


                o_tone_scores_file = df['o_tone']
                r_tone_scores_file = df['r_tone']

                # Perform paired t-test for the current file
                t_stat_file, p_value_file = ttest_rel(o_tone_scores_file, r_tone_scores_file)
                o_tone_scores.extend(df['o_tone'])
                r_tone_scores.extend(df['r_tone'])
                o_tone_mapped_scores.extend(df['o_tone_mapped'])
                r_tone_mapped_scores.extend(df['r_tone_mapped'])
                error = np.sqrt(np.sqrt(np.mean((df['o_tone'] - df['r_tone'])**2)))

                # Print results for the current file
                print(f"\nT-test for {filename}:")
                print(f"T-statistic: {t_stat_file:.2f}")
                print(f"P-value: {p_value_file:.2f}")
                print(f"Error : {error:.2f}")
                print(' *************** ****************** ******************')
                o_tone_mapped_scores_file = df['o_tone_mapped']
                r_tone_mapped_scores_file = df['r_tone_mapped']

                # Perform paired t-test for the current file
                t_stat_mapped_file, p_value_mapped_file = ttest_rel(o_tone_mapped_scores_file, r_tone_mapped_scores_file)

                # Print results for the current file
                print(f"\nT-test for 'o_tone_mapped' and 'r_tone_mapped' for {filename}:")
                print(f"T-statistic: {t_stat_mapped_file:.2f}")
                print(f"P-value: {p_value_mapped_file:.2f}")

                print('======================= ========================= =========================== ========================')

                print("\n")
overall_percentage_negative_i_tone_positive_o_tone = (total_negative_i_tone_positive_o_tone_count / total_values_count) * 100
overall_percentage_negative_i_tone_positive_r_tone = (total_negative_i_tone_positive_r_tone_count / total_values_count) * 100
print(f"Overall Percentage where 'i_tone' is negative while 'o_tone' is positive: {overall_percentage_negative_i_tone_positive_o_tone:.2f}")
print(f"Overall Percentage where 'i_tone' is negative while 'r_tone' is positive: {overall_percentage_negative_i_tone_positive_r_tone:.2f}")
# Calculate overall F1 score
overall_f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted') * 100

# Print overall F1 score
print(f"Overall F1 Score: {overall_f1:.2f}")

# Calculate and print confusion matrix
conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)

class_labels = [-1, 0, 1]
conf_matrix_df = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)
# print("Confusion Matrix:")
# print(conf_matrix_df)
overall_avg_o_tone = total_o_tone_sum / total_values_count
overall_avg_r_tone = total_r_tone_sum / total_values_count

# Print overall averages
print(f"Overall Average 'o_tone': {overall_avg_o_tone:.2f}")
print(f"Overall Average 'r_tone': {overall_avg_r_tone:.2f}")
# # Identify which class is confused with which
# for true_label in class_labels:
#     for pred_label in class_labels:
#         count = conf_matrix_df.loc[true_label, pred_label]
#         if count > 0 and true_label != pred_label:
#             print(f"Class {true_label} is confused with Class {pred_label}: {count} occurrences.")

# # Print classification report
# classification_rep = classification_report(all_true_labels, all_predicted_labels, target_names=class_labels)
# print("\nClassification Report:")
# print(classification_rep)
# Perform paired t-test
t_stat, p_value = ttest_rel(o_tone_scores, r_tone_scores)
error = np.sqrt(np.sqrt(np.mean((np.array(o_tone_scores) - np.array(r_tone_scores))**2)))

# Print the results
print(f"T-test across all files:")
print(f"T-statistic: {t_stat:.2f}")
print(f"P-value: {p_value:.2f}")

print(f"Error : {error:.2f}")

t_stat_mapped, p_value_mapped = ttest_rel(o_tone_mapped_scores, r_tone_mapped_scores)

# Print the results
print(f"T-test for 'o_tone_mapped' and 'r_tone_mapped' across all files:")
print(f"T-statistic: {t_stat_mapped:.2f}")
print(f"P-value: {p_value_mapped:.2f}")