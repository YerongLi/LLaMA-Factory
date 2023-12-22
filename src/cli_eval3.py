import json
import nltk
import logging
import os
import random
import tqdm
import csv
from bert_score import BERTScorer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from nltk.translate.bleu_score import sentence_bleu
from nltk.lm import MLE
from nltk.util import ngrams
from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc
from rouge import Rouge

LOGFILE='./evaloutput.log'
if os.path.exists(LOGFILE):
    # Remove the file
    os.remove(LOGFILE)
    print(f"The file {LOGFILE} has been removed.")
else:
    print(f"The file {LOGFILE} does not exist.")
rouge = Rouge()
BATCH_SIZE = 2
logging.basicConfig(
    format='%(asctime)s %(levelname)-4s - %(filename)-6s:%(lineno)d - %(message)s',
    level=logging.INFO,
    filename=LOGFILE,
    datefmt='%m-%d %H:%M:%S')
logging.info(f'Logger start: {os.uname()[1]}')
try:
    import platform
    if platform.system() != "Windows":
        import readline
except ImportError:
    print("Install `readline` for a better experience.")




class ChatDataset(Dataset):
    def __init__(self, data, tokenizer, template):
        self.data = data
        self.tokenizer = tokenizer
        self.template = template

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        instruction = record["instruction"]
        history = record["history"]
        record_type = record.get('type', 'unknown').replace('/', '').replace(' ', '')
        summary = record["summary"] if 'summary' in record else ''
        output = record["output"]

        # Encode the prompt using the provided encoding function
        input_ids, _ = self.template.encode_oneturn(
            tokenizer=chat_model.tokenizer, query=instruction, resp="", history=history, system=chat_model.template.system+f'\n{summary}'
        )

        # Convert to PyTorch tensors
        input_ids = torch.tensor(input_ids)
        output_ids = self.tokenizer.encode(output, add_special_tokens=True)

        return {
            'input_ids': input_ids,
            'output_ids': torch.tensor(output_ids),
        }

    def encode_prompt(self, query, resp, history, system):
        # You can customize this method based on your specific encoding requirements
        return self.tokenizer.encode(
            query + resp + history + system,
            add_special_tokens=True,
            truncation=True,
            max_length=512,  # Adjust max length as needed
            padding='max_length',
        )


def main():
    chat_model = ChatModel()
    # history = []
    # print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    # Load data from the file
    with open("data/police1.json", "r") as file:
        data = [json.loads(line) for line in file]
    # Initialize BLEURT
    # bleurt_scorer = bleurt.score.BleurtScorer("bleurt-base-128")

    # Initialize lists to store scores
    # import tqdm
    # import nltk
    # from nltk.translate.bleu_score import sentence_bleu
    # from rouge_score import rouge
    # import logging

    # Initialize lists to store scores
    bleu_scores = []
    dist1_scores = []
    dist2_scores = []
    perplexity_scores = []
    rouge_scores = []
    rouge_2_scores = []
    bert_scores = []
    scorer = BERTScorer(model_type='bert-base-uncased')

    # Type-wise scores
    type_scores = {}
    # Iterate through each record in the 'data' list
    # for record in tqdm.tqdm(data[:10]):

    ans = {}


    chat_dataset = ChatDataset(data, tokenizer=chat_model.tokenizer, template=chat_model.template)
    # Define batch size
    batch_size = 4

    # Create DataLoader
    chat_dataloader = DataLoader(chat_dataset, batch_size=batch_size, shuffle=True)

    # Iterate through batches
    for batch in chat_dataloader:
        input_ids = batch['input_ids']
        output_ids = batch['output_ids']
        print(input_ids)
        print(chat_model.tokenizer)
if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()