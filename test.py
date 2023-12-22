from transformers import AutoTokenizer
import torch

def pad_batch_with_tokenizer(tokenizer, input_ids_batch):
    # Tokenize the batch
    tokenized_batch = tokenizer(input_ids_batch, padding=True, truncation=True, return_tensors='pt')

    return {
        'input_ids': tokenized_batch['input_ids'],
        'attention_mask': tokenized_batch['attention_mask']
    }

# Example usage:
model_name = "/scratch/yerong/.cache/pyllama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"

unpadded_input_ids_batch = [
    [101, 2023, 2003, 1037, 2514, 1997, 1996, 2168, 102],
    [101, 2023, 2003, 1037, 2514, 1997, 1996, 2168, 102],
    [101, 2023, 2003, 1037, 2514, 1997, 1996, 2168, 102],
]

padded_batch = pad_batch_with_tokenizer(tokenizer, unpadded_input_ids_batch)

# Print the results
print("Unpadded batch input_ids:", unpadded_input_ids_batch)
print("\nPadded batch input_ids:\n", padded_batch['input_ids'])
print("\nAttention mask:\n", padded_batch['attention_mask'])

