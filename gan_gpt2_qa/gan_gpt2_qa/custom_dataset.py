## custom_dataset.py

import json

class Custom_Dataset:
    def __init__(self, data=None):
        self.data = data if data else {}

    def load_dataset(self, path: str) -> dict:
        """
        Load the custom QA dataset from the specified path and return it as a dictionary.

        Args:
        path (str): The path to the custom QA dataset.

        Returns:
        dict: The loaded dataset as a dictionary.
        """
        with open(path, 'r') as file:
            dataset = json.load(file)
        return dataset
