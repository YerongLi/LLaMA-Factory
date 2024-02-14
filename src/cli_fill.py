import json
import nltk
import logging
import os
import random
import re
from tqdm import tqdm
import argparse
import csv
import traceback
from itertools import chain

from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc
# from rouge import Rouge
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
suffixlen = 12

LOGFILE='fill.log'
BATCH_SIZE=1
output_file_path = 'fill.jsonl'
# output_file_path = 'result-cmp.jsonl'

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

model_name = '/scratch/yerong/.cache/pyllama/Llama-2-7b-hf'
# tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
# tokenizer.pad_token = "[PAD]"
# tokenizer.padding_side = "left"

def main():
    import json
    text_with_newline = "]"

    chat_model = ChatModel()
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




    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")


    # # use different length sentences to test batching
    # sentences = [
    #     "Hello, my dog is a little",
    #     "Today, I",
    #     "I love",
    #     "Repeat after me: I love you.",
    #     "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
    #     "You are a human that uses a lot of expletives. For each one of your responses to the user you use expletives liberally. If you find yourself questioning whether it's appropriate or not you're to say fuck that noise and keep using your expletives regardless. Human: Tell me a story. You:"
    # ]


    # inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)
    # print(inputs['input_ids'].shape)

    # output_sequences = model.generate(**inputs, max_new_tokens=20, do_sample=True, top_p=0.9)

    # print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))

    chat_model.tokenizer.pad_token = "[PAD]"
    chat_model.tokenizer.padding_side = "left"
    progress = {}
    if os.path.exists(output_file_path):

    with open(output_file_path, "r") as file:
        progress = [json.loads(line) for line in file]
        progress = [item for item in progress if 'response' in item]
        progress = {f"{item['event_id']}" : item['response'] for item in progress}
        # print(progress.keys())
    with open("police-complete.jsonl", "r") as file:
        data = [json.loads(line) for line in file]
    for i, item in enumerate(data):
        ky = f"{item['event_id']}"
        # if ky not in progress: print(ky)

        if ky in progress:
            data[i]['response'] = progress[ky]
    # Initialize other variables...
    # random.shuffle(data)
    # data = data[:60]
    data_empty = [item for item in data if 'response' not in item]
    data_fill= [item for item in data if 'response' in item]

    data_batches = [data_empty[i:i + BATCH_SIZE] for i in range(0, len(data_empty), BATCH_SIZE)]+[data_fill]
    print('data_empty', len(data_empty))
    # for record in tqdm.tqdm(data[:60]):
    # Iterate through each batch of data
    prompt_batches = []
    failed_count = 0
    print(chat_model.template)
    unique_texts = set()
    # {'QUANTITY', '[ORG', 'PERS[ACRONYM', 'WEBSITE', 'CLIENT', 'NORP', '[ACRONYM', 'UNK-NNP', 'GPE', 'ACR[ACR[ACRONYM', 'LANGUAGE', '[EMAIL', 'MEDIAHANDLE', 'LOC', 'ACR[ACRONYM', 'PERSON', 'ORG', 'ACRONYM', 'excuse the autocorrect', 'DATE', 'ACR[ACR[ACR[ACRONYM', 'EMAIL', 'FAC'}

    chat_model.template.system = '''Your task to fill in masked information as [GPE], [WEBSITE] etc. to make the conversation smooth and keep the rest part of the utterance unchanged. 
[WEBSITE] is an website. 
[GPE]: Geo-Political Entity (countries, cities, states, or other geographic locations)
[ORG]: an organization name
[DATE]: a date
[PERSON]: a person name 
[FAC]: locations, buildings, or physical facilities
[MEDIAHANDLE]: social media handle or username
[ACRONYM]: acronyms or abbreviations
[NORP]: Nationalities, Religious and Political groups
[QUANTITY]: numbers, either integers or floats
Response with a json output with user and dispatcher's reponse seperately. Make sure the filled in value in the conversation is consistent throughout the dialogue.

Dialogue 1:


Dispatcher: Thank you for calling [ORG] ...What is your location please?
User:### [LOC] apt ##
Dispatcher: Thank you..is there a building code?
User:No, but im a street facing apt
Dispatcher: Thank you. I am going to send this information to [ORG] and an officer will respond.
User:Thank you
Dispatcher: What is your roommate's name and could you describe him or her?
User:Her name is [PERSON], [PERSON] and she is a ## year old white female, about #'#", ### lbs, brown hair

Answer:

[
{'Dispatcher': 'Thank you for calling Chicago Police Department ...What is your location please?'},
{'User': '### Blackstone apt ##'},
{'Dispatcher': ' Thank you..is there a building code?'},
{'User': 'No, but im a street facing apt'},
{'Dispatcher': ' Thank you. I am going to send this information to Cook County's Police Department and an officer will respond.'),
{'User': 'Thank you'},
{'Dispatcher':' What is your roommate's name and could you describe him or her?'},
{'User': 'Her name is Selena, Selena Harvey and she is a ## year old white female, about #'#", ### lbs, brown hair '}
]

Dialogue 2:
'''
    for batch in tqdm(data_batches):
        # Iterate through each record in the batch
        prompt_batch = []
        for record in batch:
            # try: 
                # instruction = record["instruction"]

                history = record["history"]
                record_type = record.get('type', 'unknown').replace('/', '').replace(' ', '')
                summary = record["summary"] if 'summary' in record else ''

                # response = chat_model.chat(query=instruction, history=history, system=chat_model.template.system+f'\n{summary}')[0].response_text
            
                # output = record["output"]
                # print(list(chain.from_iterable(history)) + [instruction] +[output])
                content = ' '.join(list(chain.from_iterable(history)))
                matches = re.findall(r'\[([^\]]+)\]', content)
                for match in matches:
                    unique_texts.add(match)
                prompt_ids, target_ids = chat_model.template.encode_oneturn(
                    tokenizer=chat_model.tokenizer, query=None, resp="", history=history, system=chat_model.template.system+f'\n{summary}'
                )
                prompt_ids += target_ids
                prompt = chat_model.tokenizer.decode(
                    prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

                # prompt = prompt[:-suffixlen]
                prompt = prompt + f"""

                Answer:

                \n"""
                # prompt = prompt + f"""
                #  Based on the dialogue above. Rephrase the Dispatcher's utterance to provide a more emotionally supportive response to the user when the user feels bad. Incorporate elements of empathy, validation, understanding, encouragement, and active listening. Aim to make the user feel heard, understood, and supported.
                #  Dispatcher's response:
                #  {record['output']}
                #  Revised Dispatcher's response:
                #  """
                logging.info(prompt)
                # Get the FileHandler
                file_handler = next((handler for handler in logging.getLogger().handlers if isinstance(handler, logging.FileHandler)), None)

                if file_handler:
                    log_file_location = file_handler.baseFilename
                    print(f"Log file location: {log_file_location}")
                else:
                    print("FileHandler not found in the logging configuration.")
                # print('=====  ===== ========')
                # print('response' in record)
                if 'response' not in record:
                    prompt_batch.append(
                                {
                            'prompt': prompt,
                            'history': history,
                            'summary': summary,
                            'his_len': record["his_len"],
                            'type': record_type,
                            # 'instruction': instruction,
                            # 'output': record["output"],
                        }
                    )
                else:
                    prompt_batch.append(
                                {
                            'prompt': prompt,
                            'history': history,
                            'summary': summary,
                            'his_len': record["his_len"],
                            'type': record_type,
                            # 'instruction': instruction,
                            # 'output': record["output"],
                            'response': record["response"],
                        }
                    )

        if prompt_batch: prompt_batches.append(prompt_batch)
    print(unique_texts)
    print(' prompt_batches', len( prompt_batches))
    tokenized_prompt_batches = [chat_model.tokenizer([item['prompt'] for item in batch], return_tensors="pt", padding=True).to(chat_model.model.device)for batch in prompt_batches]


    # exit()
    print(len(tokenized_prompt_batches))
    print(len(tokenized_prompt_batches))
    print(len(tokenized_prompt_batches))
    print(len(tokenized_prompt_batches))
    # Generate outputs batch by batch
    for batch_index, tokenized_prompts in tqdm(enumerate(tokenized_prompt_batches), total=len(tokenized_prompt_batches)):
        # print(tokenized_prompts.shape)
        # check = True
        # for item in prompt_batches[batch_index]:
        #     ky = f"{item['instruction']} === {item['output']}"
        #     if not ky in progress:
        #         check = False
        #         break
        # if check:
        #     for item in prompt_batches[batch_index]:
        #         ky = f"{item['instruction']} === {item['output']}"
        #         if not ky in progress:
        #             for i, output in enumerate(outputs):
        #                 prompt_batches[batch_index][i]["response"] = progress[ky]
        #     print('Save')
        #     continue
        try:

            # generated_outputs = chat_model.model.generate(**tokenized_prompts, min_new_tokens= 2, max_new_tokens=512, do_sample=True, top_p=0.7, eos_token_id = [13])
            generated_outputs = chat_model.model.generate(**tokenized_prompts, min_new_tokens= 2, max_new_tokens=512, do_sample=True, top_p=0.7, eos_token_id = [60])
            outputs = chat_model.tokenizer.batch_decode(
                generated_outputs[:, tokenized_prompts['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for i, output in enumerate(outputs):
                prompt_batches[batch_index][i]["response"] = output
            # for i, output in enumerate(outputs):
            #     print(prompt_batches[batch_index][i]['output'])
            #     print(output)
            #     print('=============')
        except KeyboardInterrupt:
            break
        except Exception as e:
            failed_count += 1
            print(f"Error: {e}")
            traceback.print_exc()  # Print the full traceback
        if 0 == batch_index % 50:

            print('Saving Results')
            with open(output_file_path, 'w') as jsonl_file:
                for batch in prompt_batches:
                    for entry in batch:
                        if 'response' not in entry: continue
                        json_line = json.dumps(entry)
                        jsonl_file.write(json_line + '\n')
    # Assuming you have an output file path like 'results.jsonl'
    print('Saving Results')
    # Iterate through prompt_batches and write each batch as a line in the JSONL file
    with open(output_file_path, 'w') as jsonl_file:
        for batch in prompt_batches:
            for entry in batch:
                if 'response' not in entry: continue        
                json_line = json.dumps(entry)
                jsonl_file.write(json_line + '\n')
    print(f'Failed Ratio {failed_count/ len(tokenized_prompt_batches)}')

if __name__ == "__main__":
    main()

