from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from sacrebleu import corpus_bleu
from rouge import Rouge
import nltk

# Download the Punkt tokenizer
nltk.download("punkt")

# Load the WMT16 dataset for English-German translation
dataset = load_dataset("wmt16", "de-en")

# Define your model and tokenizer
model_name = "/scratch/yerong/.cache/pyllama/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Define metrics
bleu_metric = corpus_bleu
rouge_metric = Rouge()

# Define evaluation function
def evaluate_predictions(eval_predictions, metric_bleu, metric_rouge, tokenizer):
    # Convert predictions to text
    decoded_preds = tokenizer.batch_decode(eval_predictions, skip_special_tokens=True)

    # Load reference data
    references = dataset["validation"]["translation"]

    # Calculate BLEU score
    bleu_score = metric_bleu(decoded_preds, [references]).score

    # Calculate ROUGE score
    rouge_scores = metric_rouge.get_scores(decoded_preds, references, avg=True)

    return {"bleu": bleu_score, "rouge": rouge_scores}

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./output",
    per_device_eval_batch_size=8,
    eval_steps=500,  # Adjust based on your needs
    # Add other training arguments as needed
)

# Create a dummy trainer for evaluation only
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=None,  # No need to pass training dataset for evaluation only
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=lambda p: evaluate_predictions(p.predictions, bleu_metric, rouge_metric, tokenizer),
)

# Perform evaluation only
evaluation_metrics = trainer.evaluate()

# Access the evaluation metrics
bleu_score = evaluation_metrics["bleu"]
rouge_scores = evaluation_metrics["rouge"]

print(f"BLEU Score: {bleu_score}")
print(f"ROUGE Scores: {rouge_scores}")

# Save scores to a file if needed
with open("evaluation_scores.txt", "w") as file:
    file.write(f"BLEU Score: {bleu_score}\n")
    file.write(f"ROUGE Scores: {rouge_scores}\n")
