python src/cli_eval5.py \
    --model_name_or_path /scratch/yerong/.cache/pyllama/Llama-2-7b-hf \
    --template dispatcher \
    --finetuning_type lora \
    --quantization_bit 4 \
    --checkpoint_dir bakpolice4/best