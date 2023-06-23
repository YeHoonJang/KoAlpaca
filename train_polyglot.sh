python train.py \
    --model_name_or_path "EleutherAI/polyglot-ko-1.3b" \
    --data_path "/home/yehoon/workspace/llama_prj/data/ai_hub.json" \
    --bf16 True \
    --output_dir ./Polyglot-Ko \
    --num_train_epochs 3 \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 512 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 2000 \
    --tf32 True
#    --fsdp "full_shard auto_wrap" \
#    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \

