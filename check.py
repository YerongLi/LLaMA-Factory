import os
import csv

csv_directory = 'csv_files'

def read_and_print_history(file_path):
    with open(file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            history = row.get('history', '')
            print(f'History for row in {file_path}: {history}')

def read_all_csv_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            read_and_print_history(file_path)

if __name__ == "__main__":
    read_all_csv_files(csv_directory)
