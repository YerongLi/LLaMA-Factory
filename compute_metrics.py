import json
import nltk
import logging
import os
import random
import tqdm
import csv
from nltk.lm import MLE
from nltk.util import ngrams
from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.lm import MLE
from nltk.util import ngrams
from rouge import Rouge

# Set threshold values
bleu_threshold = 0.0
dist1_threshold = 0.0
dist2_threshold = 0.0
rouge_threshold = 0.0
rouge_2_threshold = 0.0
bert_threshold = 0.0

# Initialize language model
lm = MLE(3)

# Initialize perplexity scores
perplexity_scores = []

LOGFILE = './evaloutput.log'
if os.path.exists(LOGFILE):
    # Remove the file
    os.remove(LOGFILE)
    print(f"The file {LOGFILE} has been removed.")
else:
    print(f"The file {LOGFILE} does not exist.")

rouge = Rouge()
BATCH_SIZE = 16

logging.basicConfig(
    format='%(asctime)s %(levelname)-4s - %(filename)-6s:%(lineno)d - %(message)s',
    level=logging.INFO,
    filename=LOGFILE,
    datefmt='%m-%d %H:%M:%S'
)

logging.info(f'Logger start: {os.uname()[1]}')

try:
    import platform

    if platform.system() != "Windows":
        import readline
except ImportError:
    print("Install `readline` for a better experience.")


# Load pre-trained BERT model and tokenizer for text classification
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

def calculate_perplexity(sentence):
    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)

    # Get the model's predicted probabilities
    outputs = model(**inputs, return_dict=True)
    logits = outputs.logits

    # Calculate perplexity-like score
    perplexity_score = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))  # Assuming binary classification

    return perplexity_score.item()

# Example usage
def main():
    with open("results1.jsonl", "r") as file:
        data = [json.loads(line) for line in file]

    bleu_scores = []
    dist1_scores = []
    dist2_scores = []
    perplexity_scores = []
    rouge_scores = []
    rouge_2_scores = []
    bert_scores = []

    scorer = BERTScorer(model_type='bert-base-uncased')

    type_scores = {}
    type_perplexity_scores = {}

    ans = {}
    type_set = set()

    for record in tqdm.tqdm(data):
        instruction = record["instruction"]

        if "response" not in record:
            continue

        history = record["history"]
        record_type = record.get('type', 'unknown').replace('/', '').replace(' ', '')
        summary = record["summary"] if 'summary' in record else ''
        response = record["response"] if 'response' in record else ''

        response_tokens = nltk.word_tokenize(response)
        
        try:
            perplexity = calculate_perplexity(response)
            perplexity_scores.append(perplexity)
            type_perplexity_scores.setdefault(record_type, []).append(perplexity)
        except:
            pass

        output = record["output"]
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

        type_set.add(record_type)

        if record_type not in ans:
            ans[record_type] = []
        ans[record_type].append(response_output_pair)

        try:
            rouge_score = rouge.get_scores(response, output)
        except:
            continue
        if len(output) <= 1:
            continue

        P, R, F1 = scorer.score([response], [output])
        bert_score = F1.mean().item()
        bert_scores.append(bert_score)

        response_tokens = nltk.word_tokenize(response)
        output_tokens = nltk.word_tokenize(output)

        bleu = sentence_bleu([output_tokens], response_tokens)
        bleu_scores.append(bleu)

        if len(response_tokens) > 0:
            response_dist1 = len(set(response_tokens)) / len(response_tokens)
            response_dist2 = len(set(nltk.ngrams(response_tokens, 2))) / len(response_tokens)

        dist1_scores.append(response_dist1)
        dist2_scores.append(response_dist2)

        rouge_scores.append(rouge_score[0]['rouge-l']['f'])
        rouge_2_scores.append(rouge_score[0]['rouge-2']['f'])

        if record_type not in type_scores:
            type_scores[record_type] = {
                'bleu': [],
                'dist1': [],
                'dist2': [],
                'rouge': [],
                'rouge_2': [],
                'bert': [],
                'ppl': [],
            }

        type_scores[record_type]['bleu'].append(bleu)
        type_scores[record_type]['dist1'].append(response_dist1)
        type_scores[record_type]['dist2'].append(response_dist2)
        type_scores[record_type]['rouge'].append(rouge_score[0]['rouge-l']['f'])
        type_scores[record_type]['rouge_2'].append(rouge_score[0]['rouge-2']['f'])
        type_scores[record_type]['bert'].append(bert_score)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_dist1 = sum(dist1_scores) / len(dist1_scores)
    avg_dist2 = sum(dist2_scores) / len(dist2_scores)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    avg_rouge_2 = sum(rouge_2_scores) / len(rouge_2_scores)
    avg_bert_score = sum(bert_scores) / len(bert_scores)

    logging.info(f"Average BLEU Score (Macro): {avg_bleu * 100:.2f}")
    logging.info(f"Average DIST-1 Score (Macro): {avg_dist1 * 100:.2f}")
    logging.info(f"Average DIST-2 Score (Macro): {avg_dist2 * 100:.2f}")
    logging.info(f"Average ROUGE-L Score (Macro): {avg_rouge * 100:.2f}")
    logging.info(f"Average ROUGE-2 Score (Macro): {avg_rouge_2 * 100:.2f}")
    logging.info(f"Average BERTScore: {avg_bert_score * 100:.2f}")

    # Calculate and log average perplexity for each type
    for record_type, scores in type_perplexity_scores.items():
        avg_perplexity_type = sum(scores) / len(scores)
        type_scores[record_type]['ppl'] = avg_perplexity_type  # Store average perplexity in type_scores
        logging.info(f"Average Perplexity for Type {record_type}: {avg_perplexity_type:.2f}")

    # Calculate and log overall average perplexity
    avg_perplexity_overall = sum(perplexity_scores) / len(perplexity_scores)
    logging.info(f"Overall Average Perplexity: {avg_perplexity_overall:.2f}")

    type_set = ['SuspiciousActivity', 'AccidentTrafficParking', 'DrugsAlcohol', 'EmergencyMessage',
                'FacilitiesMaintenance', 'HarassmentAbuse', 'MentalHealth', 'NoiseDisturbance', 'TheftLostItem']

    for record_type in type_set:
        scores = type_scores[record_type]
        avg_bleu_type = sum(scores['bleu']) / len(scores['bleu'])
        avg_dist1_type = sum(scores['dist1']) / len(scores['dist1'])
        avg_dist2_type = sum(scores['dist2']) / len(scores['dist2'])
        avg_rouge_type = sum(scores['rouge']) / len(scores['rouge'])
        avg_rouge_2_type = sum(scores['rouge_2']) / len(scores['rouge_2'])
        avg_bert_type = sum(scores['bert']) / len(scores['bert'])

        logging.info(f"\nType: {record_type} < {len(scores['bleu'])}")
        logging.info(f"Average BLEU Score: {avg_bleu_type * 100:.2f}")
        logging.info(f"Average DIST-1 Score: {avg_dist1_type * 100:.2f}")
        logging.info(f"Average DIST-2 Score: {avg_dist2_type * 100:.2f}")
        logging.info(f"Average ROUGE-L Score: {avg_rouge_type * 100:.2f}")
        logging.info(f"Average ROUGE-2 Score: {avg_rouge_2_type * 100:.2f}")
        logging.info(f"Average BERTScore: {avg_bert_type * 100:.2f}")

    for record_type, pairs in ans.items():
        filename = f'{record_type}.json'

        if os.path.exists(filename):
            os.remove(filename)

        with open(filename, 'w') as f:
            json.dump(pairs, f)

    csv_header = ['instruction', 'response', 'output', 'prompt', 'history', 'summary', 'his_len', 'type']
    csv_directory = 'csv_files'
    os.makedirs(csv_directory, exist_ok=True)
    csv_writers = {}

    for record_type in ans.keys():
        csv_filename = os.path.join(csv_directory, f'{record_type}.csv')
        csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
        csv_writers[record_type] = csv.writer(csv_file)
        csv_writers[record_type].writerow(csv_header)

        os.remove(csv_filename)

    csv_all_filename = os.path.join(csv_directory, 'all.csv')
    csv_all_file = open(csv_all_filename, 'w', newline='', encoding='utf-8')
    csv_all_writer = csv.writer(csv_all_file)
    csv_all_writer.writerow(csv_header)

    for record_type, pairs in ans.items():
        for pair in pairs:
            csv_writers[record_type].writerow([pair[key] for key in csv_header])
            csv_all_writer.writerow([pair[key] for key in csv_header])

    csv_all_file.close()


if __name__ == "__main__":
    main()
