import json
import nltk
import logging
import os
import random
import tqdm
import torch
import csv
from bert_score import BERTScorer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import load_dataset

from nltk.translate.bleu_score import sentence_bleu
from nltk.lm import MLE
from nltk.util import ngrams
from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc
from rouge import Rouge
from transformers import AutoTokenizer
from functools import partial

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
            tokenizer=self.tokenizer, query=instruction, resp="", history=history, system=self.template.system+f'\n{summary}'
        )

        # Convert to PyTorch tensors
        input_ids = torch.tensor(input_ids)
        output_ids = self.tokenizer.encode(output, add_special_tokens=True)

        return {
            'input_ids': input_ids,
            'output_ids': torch.tensor(output_ids),
        }



def main():
    chat_model = ChatModel()
    tokenizer = chat_model.tokenizer
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")


    def create_prompt_formats(sample):
        """
        Format various fields of the sample ('instruction', 'context', 'response')
        Then concatenate them using two newline characters 
        :param sample: Sample dictionnary
        """

        INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        INSTRUCTION_KEY = "### Instruction:"
        INPUT_KEY = "Input:"
        RESPONSE_KEY = "### Response:"
        END_KEY = "### End"
        
        blurb = f"{INTRO_BLURB}"
        instruction = f"{INSTRUCTION_KEY}\n{sample['instruction']}"
        input_context = f"{INPUT_KEY}\n{sample['context']}" if sample["context"] else None
        response = f"{RESPONSE_KEY}\n{sample['response']}"
        end = f"{END_KEY}"
        
        parts = [part for part in [blurb, instruction, input_context, response, end] if part]

        formatted_prompt = "\n\n".join(parts)
        
        sample["text"] = formatted_prompt

        return sample
    def get_max_length(model):
        conf = model.config
        max_length = None
        for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
            max_length = getattr(model.config, length_setting, None)
            if max_length:
                print(f"Found max lenth: {max_length}")
                break
        if not max_length:
            max_length = 1024
            print(f"Using default max length: {max_length}")
        return max_length


    def preprocess_batch(batch, tokenizer, max_length):
        """
        Tokenizing a batch
        """
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
        )


    # SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
    def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, dataset: str, seed = 101):
        """Format & tokenize it so it is ready for training
        :param tokenizer (AutoTokenizer): Model Tokenizer
        :param max_length (int): Maximum number of tokens to emit from tokenizer
        """
        
        # Add prompt to each sample
        print("Preprocessing dataset...")
        dataset = dataset.map(create_prompt_formats)#, batched=True)
        
        # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
        _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
        dataset = dataset.map(
            _preprocessing_function,
            batched=True,
            remove_columns=["instruction", "context", "response", "text", "category"],
        )

        # Filter out samples that have input_ids exceeding max_length
        dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
        
        # Shuffle dataset
        dataset = dataset.shuffle(seed=seed)

        return dataset

    max_length = 1000
    dataset = preprocess_dataset(tokenizer, max_length, dataset)

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()