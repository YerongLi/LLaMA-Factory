import datasets
from transformers import pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

split_name = "test"

dataset_name, dataset_config_name = "go_emotions", "simplified"
dataset_dict = datasets.load_dataset(dataset_name, dataset_config_name)
dataset_dict[split_name][0]

# classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
classifier = pipeline(task="text-classification", model="SchuylerH/bert-multilingual-go-emtions", top_k=None)
print(dataset_dict[split_name]["text"][0])

model_outputs = classifier(dataset_dict[split_name]["text"]) 

# print(dataset_dict[split_name]["text"][0])
print(model_outputs[0])
