python src/function.py \
    --model_name_or_path $CHATLM \
    --template dispatcher \
    --finetuning_type lora \
    --quantization_bit 4 \
    --checkpoint_dir emopolice4/best