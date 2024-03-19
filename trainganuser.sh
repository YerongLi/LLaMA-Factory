# deepspeed --num_gpus 3 --master_port=9901 src/train_bash.py \
    # --deepspeed ds.json \
python src/train_bash.py \
    --stage gan \
    --model_name_or_path $CHATLM \
    --do_train \
    --dataset user \
    --template user \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --quantization_bit 4 \
    --output_dir user4 \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 10 \
    --save_total_limit 5 \
    --learning_rate 4e-4 \
    --num_train_epochs 1000.0