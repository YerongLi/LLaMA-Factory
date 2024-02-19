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
def construct_prompt(template_name, history, summary):
    tokenizer = AutoTokenizer.from_pretrained(os.environ['CHATLM'], padding_side="left")
    
    template=get_template_and_fix_tokenizer(name='user', tokenizer=tokenizer)
    prompt_ids, _ = template.encode_oneturn(
        tokenizer=tokenizer, query=None, resp=None, history=history, system=template.system+f'\n{summary}'
    )
    prompt = tokenizer.decode(
        prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    # logging.info(prompt)
    return prompt
def main():
    import json
    history = [['User', 'Hello, I need a ride to metro center, can somebody help me.'], ['Dispatcher', 'NCR Test Of the NCR Centers - Non Emergency'], ['User', 'Test received']]
    print(construct_prompt('user',history, ''))


          
if __name__ == "__main__":
    main()

