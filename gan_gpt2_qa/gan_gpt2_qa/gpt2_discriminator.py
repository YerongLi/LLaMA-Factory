## gpt2_discriminator.py

from transformers import TFGPT2Model, GPT2Tokenizer

class GPT2_Discriminator:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = TFGPT2Model.from_pretrained(model_name)

    def discriminate_question_answer_pair(self, question: str, answer: str) -> float:
        inputs = self.tokenizer(question, answer, return_tensors="tf", padding=True, truncation=True, max_length=512, return_attention_mask=True)
        outputs = self.model(inputs)
        logits = outputs.logits
        return float(logits[0][0])  # Assuming the output is a single value representing the confidence score
