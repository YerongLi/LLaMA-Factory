import json
import nltk
import logging
import os
import random
import tqdm
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

    # Iterate through each record in the 'data' list
    for record in tqdm.tqdm(data):
        instruction = record["instruction"]
        logging.info('Summary')
        logging.info(record["summary"])
        logging.info(record["history"])
        history = record["history"]
        record_type = record.get('type', 'unknown')

        response = chat_model.chat(query=instruction, history=history, system=chat_model.template.system+f'\n{record["summary"]}')[0].response_text
        output = record["output"]

        # Calculate BERTScore
        P, R, F1 = scorer.score([response], [output])
        bert_score = F1.mean()
        bert_scores.append(bert_score)

        # Logging information
        logging.info(" ===== Question ==== ")
        logging.info(instruction)
        logging.info('====   Correct =====')
        logging.info(output)
        logging.info('====   Response ==== ')
        logging.info(response)
        logging.info(f'BERTScore: {bert_score}')

        # Tokenize the sentences for BLEU and perplexity
        response_tokens = nltk.word_tokenize(response)
        output_tokens = nltk.word_tokenize(output)

        # ... (remaining code for other metric calculations)

        # Create a dictionary with the response and output pair
        response_output_pair = {
            'response': response,
            'output': record["output"],
            'prompt': prompt,
            'bert_score': bert_score
        }

        # Append the pair to the corresponding record type list in the dictionary
        if record_type not in ans:
            ans[record_type] = []
        ans[record_type].append(response_output_pair)

    # Calculate average BERTScore
    avg_bert_score = sum(bert_scores) / len(bert_scores)
    logging.info(f"Average BERTScore: {round(avg_bert_score * 100, 2)}")

    # ... (remaining code for logging average scores and type-wise scores)

    # Save response and output pairs along with BERTScore to JSON files
    for record_type, pairs in ans.items():
        filename = f'{record_type}.json'

        # Remove the original file if it exists
        if os.path.exists(filename):
            os.remove(filename)

        with open(filename, 'w') as f:
            json.dump(pairs, f)

if __name__ == "__main__":
    main()
