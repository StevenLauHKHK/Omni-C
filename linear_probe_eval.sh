########################################################################################################################
# Linear Probe Classification Evaluation Script for image modality #
# add --eval # Only perform evaluation on the linear probe model
# add --resume-linear-probe-checkpoint # Path to the linear probe checkpoint to be evaluated
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=28531 \
        --use_env linear_probe.py \
        --world_size 1 \
        --eval \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/Cars \
        --image-dataset Cars \
        --selected-multimodality image \
        --log-file linear_probe_eval_checkpoints/train_pretrained_Omni-C_image_audio_text_linear_probe_Cars_bs128_ddp.txt \
        --resume-linear-probe-checkpoint /data2/steven/MAE_25072025/linear_probe_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_linear_probe_cars_bs128_ddp/checkpoint-best.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 1 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 196