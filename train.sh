# deepspeed --num_gpus 3 --master_port=9901 src/train_bash.py \
    # --deepspeed ds.json \
python src/train_bash.py \
    --stage sft \
    --model_name_or_path /scratch/yerong/.cache/pyllama/Llama-2-7b-hf \
    --do_train \
    --dataset police \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir police2 \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 10 \
    --learning_rate 5e-5 \
    --num_train_epochs 1000.0 \
    --fp16
