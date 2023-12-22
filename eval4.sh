python src/cli_eval3.py \
    --model_name_or_path /scratch/yerong/.cache/pyllama/Llama-2-7b-hf \
    --template dispatcher \
    --finetuning_type lora \
    --quantization_bit 4 \
    --checkpoint_dir police4/best