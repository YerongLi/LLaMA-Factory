## gpt2_model.py

import torch
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config

class GPT2Model:
    def __init__(self, model_name: str = 'gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.config = GPT2Config.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name, config=self.config)

    @classmethod
    def from_pretrained(cls, model_name: str):
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        config = GPT2Config.from_pretrained(model_name)
        model = GPT2Model.from_pretrained(model_name, config=config)
        return cls(model_name, tokenizer, config, model)
