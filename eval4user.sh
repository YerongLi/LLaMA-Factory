python src/cli_eval3.py \
    --model_name_or_path $CHATLM \
    --template user \
    --dataset usertest \
    --finetuning_type lora \
    --quantization_bit 4 \
    --checkpoint_dir user4/checkpoint-1840