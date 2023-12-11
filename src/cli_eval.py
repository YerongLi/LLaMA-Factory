import json
import nltk
import logging
import os
import tqdm

from nltk.translate.bleu_score import sentence_bleu
from nltk.lm import MLE
from nltk.util import ngrams
from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc
from rouge import Rouge
rouge = Rouge()
logging.basicConfig(
    format='%(asctime)s %(levelname)-4s - %(filename)-6s:%(lineno)d - %(message)s',
    level=logging.INFO,
    filename='./output.log',
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

    # Type-wise scores
    type_scores = {}
    # Iterate through each record in the 'data' list
    # for record in tqdm.tqdm(data[:10]):


    # Assuming data is a list of records, similar to your provided code
    batch_size = 32  # Adjust as needed
    num_batches = (len(data) + batch_size - 1) // batch_size

    for batch_index in tqdm.tqdm(range(num_batches)):
        start_index = batch_index * batch_size
        end_index = (batch_index + 1) * batch_size
        batch_data = data[start_index:end_index]

        # Initialize lists to store batch-specific information
        batch_gen_kwargs = []
        # batch_prompt_lengths = []

        for record in batch_data:
            instruction = record["instruction"]
            logging.info('Summary')
            logging.info(record["summary"])
            logging.info(record["history"])
            history = [['', record["summary"]]] + record["history"]

            # Assuming self._process_args returns gen_kwargs and prompt_length
            gen_kwargs, prompt_length = chat_model._process_args(instruction, history, None)

            batch_gen_kwargs.append(gen_kwargs)
            # batch_prompt_lengths.append(prompt_length)


    for record in tqdm.tqdm(data):
        instruction = record["instruction"]
        logging.info('Summary')
        logging.info(record["summary"])
        logging.info(record["history"])
        history = [['', record["summary"]]] + record["history"]

        response = chat_model.chat(query=instruction, history=history)[0].response_text
    #     model = chat_model.model
    #     logging.info(type(model))
    #     output = record["output"]
    #     try:
    #         rouge_score = rouge.get_scores(response, output)
    #     except:
    #         continue
    #     if len(output) <= 1:
    #         continue

    #     # Logging information
    #     logging.info(" ===== Question ==== ")
    #     logging.info(instruction)
    #     logging.info('====   Correct =====')
    #     logging.info(output)
    #     logging.info('====   Response ==== ')
    #     logging.info(response)

    #     # Tokenize the sentences for BLEU and perplexity
    #     response_tokens = nltk.word_tokenize(response)
    #     output_tokens = nltk.word_tokenize(output)

    #     # Compute BLEU score
    #     bleu = sentence_bleu([output_tokens], response_tokens)
    #     bleu_scores.append(bleu)

    #     # Compute DIST-1 and DIST-2
    #     if len(response_tokens) > 0:
    #         response_dist1 = len(set(response_tokens)) / len(response_tokens)
    #         response_dist2 = len(set(nltk.ngrams(response_tokens, 2))) / len(response_tokens)

    #     dist1_scores.append(response_dist1)
    #     dist2_scores.append(response_dist2)

    #     # Compute ROUGE scores
    #     rouge_scores.append(rouge_score[0]['rouge-l']['f'])
    #     rouge_2_scores.append(rouge_score[0]['rouge-2']['f'])  # Added for ROUGE-2
    #     print(record.keys())
    #     # Store scores based on the 'type' tag
    #     record_type = record.get('type', 'unknown')
    #     if record_type not in type_scores:
    #         type_scores[record_type] = {
    #             'bleu': [],
    #             'dist1': [],
    #             'dist2': [],
    #             'rouge': [],
    #             'rouge_2': []
    #         }
    #     type_scores[record_type]['bleu'].append(bleu)
    #     type_scores[record_type]['dist1'].append(response_dist1)
    #     type_scores[record_type]['dist2'].append(response_dist2)
    #     type_scores[record_type]['rouge'].append(rouge_score[0]['rouge-l']['f'])
    #     type_scores[record_type]['rouge_2'].append(rouge_score[0]['rouge-2']['f'])

    # # Calculate average scores (using macro-averaging)
    # avg_bleu = sum(bleu_scores) / len(bleu_scores)
    # avg_dist1 = sum(dist1_scores) / len(dist1_scores)
    # avg_dist2 = sum(dist2_scores) / len(dist2_scores)
    # avg_rouge = sum(rouge_scores) / len(rouge_scores)
    # avg_rouge_2 = sum(rouge_2_scores) / len(rouge_2_scores)  # Added for ROUGE-2

    # # Log average scores (multiplied by 100 and rounded to 1 decimal place)
    # logging.info(f"Average BLEU Score (Macro): {round(avg_bleu * 100, 1)}")
    # logging.info(f"Average DIST-1 Score (Macro): {round(avg_dist1 * 100, 1)}")
    # logging.info(f"Average DIST-2 Score (Macro): {round(avg_dist2 * 100, 1)}")
    # logging.info(f"Average ROUGE-L Score (Macro): {round(avg_rouge * 100, 1)}")
    # logging.info(f"Average ROUGE-2 Score (Macro): {round(avg_rouge_2 * 100, 1)}")  # Added for ROUGE-2

    # # Print type-wise scores
    # for record_type, scores in type_scores.items():
    #     avg_bleu_type = sum(scores['bleu']) / len(scores['bleu'])
    #     avg_dist1_type = sum(scores['dist1']) / len(scores['dist1'])
    #     avg_dist2_type = sum(scores['dist2']) / len(scores['dist2'])
    #     avg_rouge_type = sum(scores['rouge']) / len(scores['rouge'])
    #     avg_rouge_2_type = sum(scores['rouge_2']) / len(scores['rouge_2'])

    #     # Log type-wise scores (multiplied by 100 and rounded to 1 decimal place)
    #     logging.info(f"\nType: {record_type} < {len(scores['bleu'])}")
    #     logging.info(f"Average BLEU Score: {round(avg_bleu_type * 100, 1)}")
    #     logging.info(f"Average DIST-1 Score: {round(avg_dist1_type * 100, 1)}")
    #     logging.info(f"Average DIST-2 Score: {round(avg_dist2_type * 100, 1)}")
    #     logging.info(f"Average ROUGE-L Score: {round(avg_rouge_type * 100, 1)}")
    #     logging.info(f"Average ROUGE-2 Score: {round(avg_rouge_2_type * 100, 1)}")


if __name__ == "__main__":
    main()
