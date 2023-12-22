import json
import nltk
import logging
import os
import random
from tqdm import tqdm
import csv
from bert_score import BERTScorer

from nltk.translate.bleu_score import sentence_bleu
from nltk.lm import MLE
from nltk.util import ngrams
from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc
from rouge import Rouge
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
LOGFILE='./evaloutput.log'
if os.path.exists(LOGFILE):
    # Remove the file
    os.remove(LOGFILE)
    print(f"The file {LOGFILE} has been removed.")
else:
    print(f"The file {LOGFILE} does not exist.")
rouge = Rouge()
BATCH_SIZE = 32
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
model_name = '/scratch/yerong/.cache/pyllama/Llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"

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
    # Load data from the file
    with open("data/police1.json", "r") as file:
        data = [json.loads(line) for line in file]

    # Initialize other variables...
    random.shuffle(data)
    # data = data[:60]
    # Group data into batches
    data_batches = [data[i:i + BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)]

    # for record in tqdm.tqdm(data[:60]):
    # Iterate through each batch of data
    prompt_batches = []
    failed_count = 0
    for batch in tqdm(data_batches):
        # Iterate through each record in the batch
        prompt_batch = []
        for record in batch:
            try: 
                instruction = record["instruction"]

                # logging.info('Summary')
                # logging.info(record["summary"])
                # logging.info(record["history"])
                history = record["history"]
                record_type = record.get('type', 'unknown').replace('/', '').replace(' ', '')
                summary = record["summary"] if 'summary' in record else ''

                # response = chat_model.chat(query=instruction, history=history, system=chat_model.template.system+f'\n{summary}')[0].response_text
            
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
                        'output': record["output"],
                        'prompt': prompt,
                        'history': history,
                        'summary': summary,
                        'his_len': record["his_len"],
                        'type': record_type,
                    }
                )
            except:
                continue

        if prompt_batch: prompt_batches.append(prompt_batch)
    
    # tokenized_prompt_batches = [[item['prompt'] for item in batch] for batch in prompt_batches]
    # print(tokenized_prompt_batches[:2])
    tokenized_prompt_batches = [chat_model.tokenizer([item['prompt'] for item in batch], return_tensors="pt", padding=True).to(chat_model.model.device)for batch in prompt_batches]

    # Generate outputs batch by batch
    for tokenized_prompts in tqdm(tokenized_prompt_batches):
        # print(tokenized_prompts.shape)
        try:

            generated_outputs = chat_model.model.generate(**tokenized_prompts, max_new_tokens=20, do_sample=True, top_p=0.9)
            

            outputs = chat_model.tokenizer.batch_decode(
                generated_outputs[:, tokenized_prompts['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        # print(outputs)
        # for prompt, generated_output in zip(tokenized_prompts["input_ids"], generated_outputs):
        #     decoded_output = tokenizer.decode(generated_output, skip_special_tokens=True)
        #     print(f"Input: {tokenizer.decode(prompt, skip_special_tokens=True)}")
        #     print(f"Generated Output: {decoded_output}")
        #     print("=" * 50)
        except KeyboardInterrupt:
            break
        except:
            failed_count+= 1
            continue
    print(f'Failed Ratio {failed_count/ len(tokenized_prompt_batches)}')

if __name__ == "__main__":
    main()

