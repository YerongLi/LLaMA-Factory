import os
import pandas as pd

# Function to extract and process file names
def process_filenames(folder_path):
    # Get all file names in the folder
    all_files = os.listdir(folder_path)
    print(all_files)
    all_files = ['i-LIWC-22 Results - all - LIWC Analysis.csv',
    'o-LIWC-22 Results - all - LIWC Analysis.csv',
    'r-LIWC-22 Results - all - LIWC Analysis.csv',]
    # Group file names based on suffix
    groups = {}
    for name in all_files:
        prefix = name[0]
        suffix = name[2:]
        if suffix not in groups:
            groups[suffix] = {'i': None, 'o': None, 'r': None}
        groups[suffix][prefix] = name

    # Process CSV files within each group
    for suffix, files in groups.items():
        print(f"\nProcessing group for suffix '{suffix}':")

        # Create a DataFrame to store the data
        df = pd.DataFrame()

        for prefix, file_name in files.items():
            if file_name is not None and file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                file_df = pd.read_csv(file_path)
                if prefix == 'i':
                    # Extract columns and save them in the DataFrame
                    df[f"instruction"] = file_df['instruction']
                    df[f"response"] = file_df['response']
                    df[f"output"] = file_df['output']
                    df[f"prompt"] = file_df['prompt']
                    df[f"history"] = file_df['history']
                    df[f"summary"] = file_df['summary']
                    df[f"type"] = file_df['type']
                    try:
                        df[f"his_len"] = file_df['his_len']
                    except:
                        pass

                # Save tone with a prefix in the column name
                df[f"{prefix}_tone"] = file_df['Tone']

        # Save the combined DataFrame
        output_file_path = os.path.join('out', f"group_{suffix}.csv")
        df.to_csv(output_file_path, index=False)

# Replace 'folder_path' with the actual path to your folder
folder_path = "out_csv"
process_filenames(folder_path)
