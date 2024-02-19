import json
import nltk
import logging
import os
import random
from tqdm import tqdm
import csv
import traceback
from llmtuner.data.template import get_template_and_fix_tokenizer
# from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
LOGFILE='./evaloutput.log'
BATCH_SIZE=10

# if os.path.exists(LOGFILE):
#     # Remove the file
#     os.remove(LOGFILE)
#     print(f"The file {LOGFILE} has been removed.")
# else:
#     print(f"The file {LOGFILE} does not exist.")
# # rouge = Rouge()
# logging.basicConfig(
#     format='%(asctime)s %(levelname)-4s - %(filename)-6s:%(lineno)d - %(message)s',
#     level=logging.INFO,
#     filename=LOGFILE,
#     datefmt='%m-%d %H:%M:%S')
# logging.info(f'Logger start: {os.uname()[1]}')

# # tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
# # tokenizer.pad_token = "[PAD]"
# # tokenizer.padding_side = "left"
def construct_prompt(
    template_name: str,
    history: List[str],
    ):
    tokenizer = AutoTokenizer.from_pretrained('/scratch/yerong/.cache/pyllama/Llama-2-7b-chat-hf', padding_side="left")
    
    template=get_template_and_fix_tokenizer(name='user', tokenizer=tokenizer)
    prompt_ids, _ = template.encode_oneturn(
        tokenizer=tokenizer, query=None, resp=None, history=history, system=template.system+f'\n{summary}'
    )
    prompt = tokenizer.decode(
        prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    logging.info(prompt)
    return prompt
def main():
    import json
    history = [['User', 'Hello, I need a ride to metro center, can somebody help me.'], ['Dispatcher', 'NCR Test Of the NCR Centers - Non Emergency'], ['User', 'Test received']]
    print(construct_prompt(history))




#     # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")


#     # # use different length sentences to test batching
#     # sentences = [
#     #     "Hello, my dog is a little",
#     #     "Today, I",
#     #     "I love",
#     #     "Repeat after me: I love you.",
#     #     "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
#     #     "You are a human that uses a lot of expletives. For each one of your responses to the user you use expletives liberally. If you find yourself questioning whether it's appropriate or not you're to say fuck that noise and keep using your expletives regardless. Human: Tell me a story. You:"
#     # ]


#     # inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)
#     # print(inputs['input_ids'].shape)

#     # output_sequences = model.generate(**inputs, max_new_tokens=20, do_sample=True, top_p=0.9)

#     # print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))

#     progress = {}
    

#     # if os.path.exists(output_file_path):

#     #     with open(output_file_path, "r") as file:
#     #         progress = [json.loads(line) for line in file]
#     #         progress = [item for item in progress if 'response' in item]
#     #         progress = {f"{item['event_id']}==={item['instruction']}==={item['output']}" : item['response'] for item in progress}
#     #         # print(progress.keys())

#     # Load data from the file
#     # with open("data/police1.jsonl", "r") as file:
#     with open("data/usertest.jsonl", "r") as file:
#         data = [json.loads(line) for line in file]
#         for i, item in enumerate(data):
#             ky = f"{item['event_id']}==={item['instruction']}==={item['output']}"
#             # if ky not in progress: print(ky)

#             if ky in progress:
#                 data[i]['response'] = progress[ky]

#     data_empty = [item for item in data if 'response' not in item]
#     data_fill= [item for item in data if 'response' in item]

#     data_batches = [data_empty[i:i + BATCH_SIZE] for i in range(0, len(data_empty), BATCH_SIZE)]+[data_fill]
#     print('progress', len(progress))
#     print('data_empty', len(data_empty))
#     print('data_fill', len(data_fill))

#     # for record in tqdm.tqdm(data[:60]):
#     # Iterate through each batch of data
#     prompt_batches = []
#     failed_count = 0
#     for batch in tqdm(data_batches):
#         # Iterate through each record in the batch
#         prompt_batch = []
#         for record in batch:
#             instruction = record["instruction"]

#             # logging.info('Summary')
#             # logging.info(record["summary"])
#             # logging.info(record["history"])
#             history = record["history"]
#             record_type = record.get('type', 'unknown').replace('/', '').replace(' ', '')
#             summary = record["summary"] if 'summary' in record else ''

#             # response = chat_model.chat(query=instruction, history=history[:-1], system=chat_model.template.system+f'\n{summary}')[0].response_text
        
#             output = record["output"]

            # # prompt_ids, _ = chat_model.template.encode_oneturn(
            # #     tokenizer=chat_model.tokenizer, query=instruction, resp=None, history=history, system=chat_model.template.system+f'\n{summary}'
            # # )
            # prompt_ids, _ = template.encode_oneturn(
            #     tokenizer=tokenizer, query=instruction, resp=None, history=history, system=template.system+f'\n{summary}'
            # )
            # prompt = tokenizer.decode(
            #     prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            # )
            # logging.info(prompt)
          
if __name__ == "__main__":
    main()

