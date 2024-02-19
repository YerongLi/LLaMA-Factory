# python src/cli_eval5.py \
#     --model_name_or_path $LM \
#     --template dispatcher \
#     --finetuning_type lora \
#     --quantization_bit 4

python src/cli_eval3.py \
    --model_name_or_path $LM \
    --template dispatcher \
    --finetuning_type lora \
    --quantization_bit 4 \
    --checkpoint_dir police4/best