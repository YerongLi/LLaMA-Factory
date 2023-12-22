import json
import nltk
import logging
import os
import random
import tqdm
import csv
from bert_score import BERTScorer

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


def main():
    chat_model = ChatModel()

    # Load data from the file
    with open("data/police1.json", "r") as file:
        data = [json.loads(line) for line in file]

    # Initialize other variables...

    # Group data into batches
    data_batches = [data[i:i + BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)]

    # Iterate through each batch of data
    prompt_batches = []

    for batch in tqdm.tqdm(data_batches):
        # Iterate through each record in the batch
        prompt_batch = []
        for record in batch:
            print(record)
            instruction = record["instruction"]
            # logging.info('Summary')
            # logging.info(record["summary"])
            # logging.info(record["history"])
            history = record["history"]
            record_type = record.get('type', 'unknown').replace('/', '').replace(' ', '')
            summary = record["summary"] if 'summary' in record else ''

            if record_type in {'DrugsAlcohol', 'HarassmentAbuse', 'MentalHealth', 'TheftLostItem', 'SuspiciousActivity', 'EmergencyMessage'}: continue
            # response = chat_model.chat(query=instruction, history=history, system=chat_model.template.system+f'\n{summary}')[0].response_text
            print(record_type)
            print(record)
            logging.info(record)
            continue
            output = record["output"]

            prompt_ids, _ = chat_model.template.encode_oneturn(
                tokenizer=chat_model.tokenizer, query=instruction, resp="", history=history, system=chat_model.template.system+f'\n{summary}'
            )
            prompt = chat_model.tokenizer.decode(
                prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            prompt_batch.append(
                        {
                    'instruction': instruction,
                    'response': response,
                    'output': record["output"],
                    'prompt': prompt,
                    'history': history,
                    'summary': summary,
                    'his_len': record["his_len"],
                    'type': record_type,
                }
            )

        prompt_batches.append(prompt_batch)
        break
    print(prompt_batches[:2])
                # ... (rest of the code)

if __name__ == "__main__":
    main()

