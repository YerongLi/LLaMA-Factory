import pandas as pd
import os

# Specify the directory where the CSV files are located
directory_path = 'out'

# Initialize counters
count_greater_than_60 = {'i_tone': 0, 'r_tone': 0, 'o_tone': 0}
count_smaller_than_40 = {'i_tone': 0, 'r_tone': 0, 'o_tone': 0}

# Loop through each CSV file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # Count occurrences where 'i_tone', 'r_tone', and 'o_tone' are greater than 60
        count_greater_than_60['i_tone'] += (df['i_tone'] > 60).sum()
        count_greater_than_60['r_tone'] += (df['r_tone'] > 60).sum()
        count_greater_than_60['o_tone'] += (df['o_tone'] > 60).sum()

        # Count occurrences where 'i_tone', 'r_tone', and 'o_tone' are smaller than 40
        count_smaller_than_40['i_tone'] += (df['i_tone'] < 40).sum()
        count_smaller_than_40['r_tone'] += (df['r_tone'] < 40).sum()
        count_smaller_than_40['o_tone'] += (df['o_tone'] < 40).sum()

# Print the results
print(f"Count of 'i_tone' values greater than 60: {count_greater_than_60['i_tone']}")
print(f"Count of 'r_tone' values greater than 60: {count_greater_than_60['r_tone']}")
print(f"Count of 'o_tone' values greater than 60: {count_greater_than_60['o_tone']}")

print(f"Count of 'i_tone' values smaller than 40: {count_smaller_than_40['i_tone']}")
print(f"Count of 'r_tone' values smaller than 40: {count_smaller_than_40['r_tone']}")
print(f"Count of 'o_tone' values smaller than 40: {count_smaller_than_40['o_tone']}")
