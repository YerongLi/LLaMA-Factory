import json
import nltk
import logging
import os
import random
from tqdm import tqdm
import csv
import traceback


from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
LOGFILE='./evaloutput.log'
BATCH_SIZE=10

if os.path.exists(LOGFILE):
    # Remove the file
    os.remove(LOGFILE)
    print(f"The file {LOGFILE} has been removed.")
else:
    print(f"The file {LOGFILE} does not exist.")
# rouge = Rouge()
logging.basicConfig(
    format='%(asctime)s %(levelname)-4s - %(filename)-6s:%(lineno)d - %(message)s',
    level=logging.INFO,
    filename=LOGFILE,
    datefmt='%m-%d %H:%M:%S')
logging.info(f'Logger start: {os.uname()[1]}')

# tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
# tokenizer.pad_token = "[PAD]"
# tokenizer.padding_side = "left"

def main():
    import json
    text_with_newline = "\n"
    chat_model = ChatModel()
    logging.info('chat_model')

    logging.info(chat_model.args)
    output_file_path = f'{chat_model.args[0].split("/")[-2]}.jsonl'
    logging.info(output_file_path)
    tokens = chat_model.tokenizer.encode(text_with_newline)

    logging.info(tokens)
    # Initialize BLEURT
    # bleurt_scorer = bleurt.score.BleurtScorer("bleurt-base-128")

    # Initialize lists to store scores
    # import tqdm
    # import nltk
    # from nltk.translate.bleu_score import sentence_bleu
    # from rouge_score import rouge
    # import logging

    # Initialize lists to store scores

    # Type-wise scores
    type_scores = {}
    # Iterate through each record in the 'data' list

    ans = {}





    chat_model.tokenizer.pad_token = "[PAD]"
    chat_model.tokenizer.padding_side = "left"
    progress = {}
    

    if os.path.exists(output_file_path):

        with open(output_file_path, "r") as file:
            progress = [json.loads(line) for line in file]
            progress = [item for item in progress if 'response' in item]
            progress = {f"{item['event_id']}==={item['instruction']}==={item['output']}" : item['response'] for item in progress}
            # print(progress.keys())

    # Load data from the file
    if chat_model.args[0].split("/")[-2].startswith("user"):
        # i_file_name = "data/usertest.jsonl"
        i_file_name = "data/usertest.jsonl"
    elif chat_model.args[0].split("/")[-2].startswith("police") or chat_model.args[0].split("/")[-2].startswith("dispatcher"):
        i_file_name = "data/dispatchertest.jsonl"

    # Read the JSONL file
    with open(i_file_name, "r") as file:
        data = [json.loads(line) for line in file]
        for i, item in enumerate(data):
            ky = f"{item['event_id']}==={item['instruction']}==={item['output']}"
            # if ky not in progress: print(ky)

            if ky in progress:
                data[i]['response'] = progress[ky]

    data_empty = [item for item in data if 'response' not in item]
    data_fill= [item for item in data if 'response' in item]

    data_batches = [data_empty[i:i + BATCH_SIZE] for i in range(0, len(data_empty), BATCH_SIZE)]+[data_fill]
    print('progress', len(progress))
    print('data_empty', len(data_empty))
    print('data_fill', len(data_fill))

    # for record in tqdm.tqdm(data[:60]):
    # Iterate through each batch of data
    prompt_batches = []
    failed_count = 0
    for batch in tqdm(data_batches):
        # Iterate through each record in the batch
        prompt_batch = []
        for record in batch:
            instruction = record["instruction"]

            # logging.info('Summary')
            # logging.info(record["summary"])
            # logging.info(record["history"])
            history = record["history"]
            record_type = record.get('type', 'unknown').replace('/', '').replace(' ', '')
            summary = record["summary"] if 'summary' in record else ''

            # response = chat_model.chat(query=instruction, history=history[:-1], system=chat_model.template.system+f'\n{summary}')[0].response_text
        
            output = record["output"]

            prompt_ids, _ = chat_model.template.encode_oneturn(
                tokenizer=chat_model.tokenizer, query=instruction, resp=None, history=history, system=chat_model.template.system+f'\n{summary}'
            )
            prompt = chat_model.tokenizer.decode(
                prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            logging.info(prompt)
            if 'response' not in record:
                prompt_batch.append(
                    {
                        'event_id': record["event_id"],
                        'prompt': prompt,
                        'history': history,
                        'summary': summary,
                        'his_len': record["his_len"],
                        'type': record_type,
                        'instruction': record["instruction"],
                        'output': record["output"],
                    }
                )
            else:
                prompt_batch.append(
                    {
                        'event_id': record["event_id"],
                        'prompt': prompt,
                        'history': history,
                        'summary': summary,
                        'his_len': record["his_len"],
                        'type': record_type,
                        'instruction': record["instruction"],
                        'output': record["output"],
                        'response': record["response"],
                    }
                )

        if prompt_batch: prompt_batches.append(prompt_batch)
    
        
    tokenized_prompt_batches = [chat_model.tokenizer([item['prompt'] for item in batch], return_tensors="pt", padding=True).to(chat_model.model.device)for batch in prompt_batches[:-1]]
    print(' tokenized_prompt_batches', len( tokenized_prompt_batches))


    # exit()
    print(len(tokenized_prompt_batches))
    print(len(tokenized_prompt_batches))
    print(len(tokenized_prompt_batches))
    print(len(tokenized_prompt_batches))
    # Generate outputs batch by batch
    for batch_index, tokenized_prompts in tqdm(enumerate(tokenized_prompt_batches), total=len(tokenized_prompt_batches)):
        # print(tokenized_prompts.shape)
        try:

            generated_outputs = chat_model.model.generate(**tokenized_prompts, min_new_tokens= 2, max_new_tokens=512, do_sample=True, top_p=0.7, eos_token_id = [13])
            outputs = chat_model.tokenizer.batch_decode(
                generated_outputs[:, tokenized_prompts['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for i, output in enumerate(outputs):
                prompt_batches[batch_index][i]["response"] = output

        except KeyboardInterrupt:
            break
        except Exception as e:
            failed_count += 1
            print(f"Error: {e}")
            traceback.print_exc()  # Print the full traceback
        if 0 == batch_index % 10:

            print('Saving Results')
            with open(output_file_path, 'w') as jsonl_file:
                for batch in prompt_batches:
                    for entry in batch:
                        json_line = json.dumps(entry)
                        jsonl_file.write(json_line + '\n')
    # Assuming you have an output file path like 'results.jsonl'
    print('Saving Results')
    # Iterate through prompt_batches and write each batch as a line in the JSONL file
    with open(output_file_path, 'w') as jsonl_file:
        for batch in prompt_batches:
            for entry in batch:
                json_line = json.dumps(entry)
                jsonl_file.write(json_line + '\n')
    print(f'Failed Ratio {failed_count/ len(tokenized_prompt_batches)}')

if __name__ == "__main__":
    main()

