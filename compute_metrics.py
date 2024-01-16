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
from rouge import Rouge
bleu_threshold = 0.0
# bleu_threshold = 0.0

dist1_threshold = 0.0
# dist1_threshold = 0.7

dist2_threshold = 0.0
# dist2_threshold = 0.8

rouge_threshold = 0.0
# rouge_threshold = 0.12

rouge_2_threshold = 0.0
# rouge_2_threshold = 0.12

bert_threshold = 0.0
# bert_threshold = 0.2
LOGFILE='./evaloutput.log'
if os.path.exists(LOGFILE):
    # Remove the file
    os.remove(LOGFILE)
    print(f"The file {LOGFILE} has been removed.")
else:
    print(f"The file {LOGFILE} does not exist.")
rouge = Rouge()
BATCH_SIZE = 8
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
    # chat_model = ChatModel()
    # history = []
    # print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    # Load data from the file
    # with open("results_gpt35.jsonl", "r") as file:
    # with open("results-cmp.jsonl", "r") as file:
    with open("results1.jsonl", "r") as file:
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

    # random.shuffle(data)
    # for record in tqdm.tqdm(data[:10]):
    type_set = set()
    for record in tqdm.tqdm(data):
        instruction = record["instruction"]
        # logging.info('Summary')
        # logging.info(record["summary"])
        # logging.info(record["history"])
        if "response" not in  record: continue

        history = record["history"]
        record_type = record.get('type', 'unknown').replace('/', '').replace(' ', '')
        summary = record["summary"] if 'summary' in record else ''
        response = record["response"] if 'response' in record else ''

        # response = chat_model.chat(query=instruction, history=history, system=chat_model.template.system+f'\n{summary}')[0].response_text
        # logging.info(record)
        output = record["output"]

        # prompt_ids, _ = chat_model.template.encode_oneturn(
        #     tokenizer=chat_model.tokenizer, query=instruction, resp="", history=history, system=chat_model.template.system+f'\n{summary}'
        # )
        # prompt = chat_model.tokenizer.decode(
        #     prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        # )
        prompt = record['prompt'] if 'prompt' in record else ''

        response_output_pair = {
            'instruction': instruction,
            'response': response,
            'output': record["output"],
            'prompt': prompt,
            'history': history,
            'summary': summary,
            'his_len': record["his_len"],
            'type': record_type,
        }
        # print(record_type)
        type_set.add(record_type)
        # Append the pair to the corresponding record type list in the dictionary
        if record_type not in ans:
            ans[record_type] = []
        ans[record_type].append(response_output_pair)
        try:
            rouge_score = rouge.get_scores(response, output)
        except:
            continue
        if len(output) <= 1:
            continue


        # Calculate BERTScore
        P, R, F1 = scorer.score([response], [output])
        bert_score = F1.mean().item()
        bert_scores.append(bert_score)
        # # Logging information
        # logging.info(" ===== Question ==== ")
        # logging.info(instruction)
        # logging.info('====   Correct =====')
        # logging.info(output)
        # logging.info('====   Response ==== ')
        # logging.info(response)

        # Tokenize the sentences for BLEU and perplexity
        response_tokens = nltk.word_tokenize(response)
        output_tokens = nltk.word_tokenize(output)

        # Compute BLEU score
        bleu = sentence_bleu([output_tokens], response_tokens)
        bleu_scores.append(bleu)

        # Compute DIST-1 and DIST-2
        if len(response_tokens) > 0:
            response_dist1 = len(set(response_tokens)) / len(response_tokens)
            response_dist2 = len(set(nltk.ngrams(response_tokens, 2))) / len(response_tokens)

        dist1_scores.append(response_dist1)
        dist2_scores.append(response_dist2)

        # Compute ROUGE scores
        rouge_scores.append(rouge_score[0]['rouge-l']['f'])
        rouge_2_scores.append(rouge_score[0]['rouge-2']['f'])  # Added for ROUGE-2
        # Store scores based on the 'type' tag
        if record_type not in type_scores:
            type_scores[record_type] = {
                'bleu': [],
                'dist1': [],
                'dist2': [],
                'rouge': [],
                'rouge_2': [],
                'bert': [],
            }
        type_scores[record_type]['bleu'].append(bleu)
        type_scores[record_type]['dist1'].append(response_dist1)
        type_scores[record_type]['dist2'].append(response_dist2)
        type_scores[record_type]['rouge'].append(rouge_score[0]['rouge-l']['f'])
        type_scores[record_type]['rouge_2'].append(rouge_score[0]['rouge-2']['f'])
        type_scores[record_type]['bert'].append(bert_score)
    # Calculate average scores (using macro-averaging)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_dist1 = sum(dist1_scores) / len(dist1_scores)
    avg_dist2 = sum(dist2_scores) / len(dist2_scores)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    avg_rouge_2 = sum(rouge_2_scores) / len(rouge_2_scores)  # Added for ROUGE-2
    avg_bert_score = sum(bert_scores) / len(bert_scores)

    logging.info(f"Average BLEU Score (Macro): {avg_bleu * 100:.2f}")
    logging.info(f"Average DIST-1 Score (Macro): {avg_dist1 * 100:.2f}")
    logging.info(f"Average DIST-2 Score (Macro): {avg_dist2 * 100:.2f}")
    logging.info(f"Average ROUGE-L Score (Macro): {avg_rouge * 100:.2f}")
    logging.info(f"Average ROUGE-2 Score (Macro): {avg_rouge_2 * 100:.2f}")
    logging.info(f"Average BERTScore: {avg_bert_score * 100:.2f}")
    
    type_set = ['SuspiciousActivity', 'AccidentTrafficParking', 'DrugsAlcohol', 'EmergencyMessage', 'FacilitiesMaintenance', 'HarassmentAbuse', 'MentalHealth', 'NoiseDisturbance', 'TheftLostItem']
    # Print type-wise scores
    # for record_type, scores in type_scores.items():
    for record_type in type_set:
        scores = type_scores[record_type]
        avg_bleu_type = sum(scores['bleu']) / len(scores['bleu'])
        avg_dist1_type = sum(scores['dist1']) / len(scores['dist1'])
        avg_dist2_type = sum(scores['dist2']) / len(scores['dist2'])
        avg_rouge_type = sum(scores['rouge']) / len(scores['rouge'])
        avg_rouge_2_type = sum(scores['rouge_2']) / len(scores['rouge_2'])
        avg_bert_type = sum(scores['bert']) / len(scores['bert'])

        # Log type-wise scores (multiplied by 100 and rounded to 2 decimal places)
        logging.info(f"\nType: {record_type} < {len(scores['bleu'])}")
        logging.info(f"Average BLEU Score: {avg_bleu_type * 100:.2f}")
        logging.info(f"Average DIST-1 Score: {avg_dist1_type * 100:.2f}")
        logging.info(f"Average DIST-2 Score: {avg_dist2_type * 100:.2f}")
        logging.info(f"Average ROUGE-L Score: {avg_rouge_type * 100:.2f}")
        logging.info(f"Average ROUGE-2 Score: {avg_rouge_2_type * 100:.2f}")
        logging.info(f"Average BERTScore: {avg_bert_type * 100:.2f}")
    for record_type, pairs in ans.items():
        filename = f'{record_type}.json'

        # Remove the original file if it exists
        if os.path.exists(filename):
            os.remove(filename)

        with open(filename, 'w') as f:
            json.dump(pairs, f)


    # Define the CSV file header
    csv_header = ['instruction', 'response', 'output', 'prompt', 'history', 'summary','his_len', 'type']

    # Specify the directory for CSV files
    csv_directory = 'csv_files'

    # Create the directory if it doesn't exist
    os.makedirs(csv_directory, exist_ok=True)

    # Dictionary to store CSV writers for each record type
    csv_writers = {}

    # Open CSV files for writing based on record type
    # # anskeys= {}

    type_set = ['SuspiciousActivity', 'AccidentTrafficParking', 'DrugsAlcohol', 'EmergencyMessage', 'FacilitiesMaintenance', 'HarassmentAbuse', 'MentalHealth', 'NoiseDisturbance', 'TheftLostItem']
    for record_type in ans.keys():
        csv_filename = os.path.join(csv_directory, f'{record_type}.csv')
        csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
        csv_writers[record_type] = csv.writer(csv_file)
        csv_writers[record_type].writerow(csv_header)

        os.remove(csv_filename)


    # Open a CSV file for writing containing data from all types
    csv_all_filename = os.path.join(csv_directory, 'all.csv')
    csv_all_file = open(csv_all_filename, 'w', newline='', encoding='utf-8')
    csv_all_writer = csv.writer(csv_all_file)
    csv_all_writer.writerow(csv_header)

    # ... (existing code)

    # Write each record to the respective CSV file and the 'all.csv' file
    for record_type, pairs in ans.items():
        for pair in pairs:
            csv_writers[record_type].writerow([pair[key] for key in csv_header])
            csv_all_writer.writerow([pair[key] for key in csv_header])

    # Close all CSV files
    # for writer in csv_writers.values():
    #     writer.file.close()

    csv_all_file.close()
if __name__ == "__main__":
    main()
