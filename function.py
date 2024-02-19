import json
import nltk
import logging
import os
import csv
from transformers import AutoTokenizer
# from llmtuner.data.template import get_template_and_fix_tokenizer
from src import get_template_and_fix_tokenizer
import torch
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
    print(construct_prompt('user',history, 'Yiren summary'))


          
if __name__ == "__main__":
    main()

