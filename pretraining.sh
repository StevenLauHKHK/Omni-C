CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port=29503 \
        --use_env pretraining.py \
        --world_size 8 \
        --simclr \
        --image-data-path /data_ssd/DATA/imagenet1k \
        --image-dataset imagenet \
        --audio-data-path /data_ssd/DATA/audioset \
        --audio-dataset audioset \
        --text-data-path /data_ssd/DATA/Wikipedia \
        --text-dataset wiki \
        --selected-multimodality image audio text \
        --model ViT-B-32 \
        --model-size base \
        --checkpoint-dir pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector \
        --debug-dir pretrained_checkpoints/debug_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector \
        --log-file pretrained_checkpoints/train_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector.log \
        --batch-size 32 \
        --accum-iter 1 \
        --save_every_n 5 \
        --opts TRAIN.EPOCHS 100 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-6 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 5 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUG.PRESET simclr DATA.IMG_SIZE 224 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50