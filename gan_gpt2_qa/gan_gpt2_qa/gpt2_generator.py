## gpt2_generator.py

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2_Generator:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_question_answer_pair(self):
        # Generate a question-answer pair using the GPT-2 model
        input_text = "Generate question: "
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=50, num_return_sequences=1)
        generated_question = self.tokenizer.decode(output[0], skip_special_tokens=True)

        input_text = "Generate answer for the question: " + generated_question
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=50, num_return_sequences=1)
        generated_answer = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return generated_question, generated_answer
