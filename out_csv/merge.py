import os

# Function to extract and process file names
def process_filenames(folder_path):
    # Get all file names in the folder
    all_files = os.listdir(folder_path)

    # Extract and print the set of file names without prefixes
    file_set = set(name[2:] for name in all_files)
    print("Set of file names without prefixes:", file_set)

    # Group file names based on suffix
    groups = {}
    for name in all_files:
        prefix = name[0]
        suffix = name[2:]
        if suffix not in groups:
            groups[suffix] = {'i': None, 'o': None, 'r': None}
        groups[suffix][prefix] = name

    # Print the groups
    for suffix, files in groups.items():
        print(f"Group for suffix '{suffix}':")
        for prefix, file_name in files.items():
            print(f"{prefix}_{suffix}: {file_name}")

# Replace 'folder_path' with the actual path to your folder
folder_path = "."
process_filenames(folder_path)
