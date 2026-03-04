###############################################################################
# SBoRA Fine Tuning Script for image modality #

# Car dataset SBoRA fine tuning with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29412 \
        --use_env fine_tune_sbora.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/Cars \
        --image-dataset Cars \
        --selected-multimodality image \
        --checkpoint-dir fine_tune_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_image_fine_tune_Cars_bs128_ddp \
        --log-file fine_tune_checkpoints/train_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_image_fine_tune_Cars_bs128_ddp.log \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --sbora-rank 128 \
        --sbora-alpha 128 \
        --sbora-dropout 0.0 \
        --sbora-mode FA \
        --freeze-patch-embedding \
        --sbora-base-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 196

# DTD dataset SBoRA fine tuning with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29412 \
        --use_env fine_tune_sbora.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/DTD \
        --image-dataset DTD \
        --selected-multimodality image \
        --checkpoint-dir fine_tune_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_image_fine_tune_DTD_bs128_ddp \
        --log-file fine_tune_checkpoints/train_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_image_fine_tune_DTD_bs128_ddp.log \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --sbora-rank 128 \
        --sbora-alpha 128 \
        --sbora-dropout 0.0 \
        --sbora-mode FA \
        --freeze-patch-embedding \
        --sbora-base-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 47

# EuroSAT dataset SBoRA fine tuning with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29412 \
        --use_env fine_tune_sbora.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/EuroSAT \
        --image-dataset EuroSAT \
        --selected-multimodality image \
        --checkpoint-dir fine_tune_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_image_fine_tune_EuroSAT_bs128_ddp \
        --log-file fine_tune_checkpoints/train_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_image_fine_tune_EuroSAT_bs128_ddp.log \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --sbora-rank 128 \
        --sbora-alpha 128 \
        --sbora-dropout 0.0 \
        --sbora-mode FA \
        --freeze-patch-embedding \
        --sbora-base-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 10

# GTSRB dataset SBoRA fine tuning with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29412 \
        --use_env fine_tune_sbora.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/GTSRB \
        --image-dataset GTSRB \
        --selected-multimodality image \
        --checkpoint-dir fine_tune_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_image_fine_tune_GTSRB_bs128_ddp \
        --log-file fine_tune_checkpoints/train_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_image_fine_tune_GTSRB_bs128_ddp.log \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --use-sbora \
        --sbora-rank 128 \
        --sbora-alpha 128 \
        --sbora-dropout 0.0 \
        --sbora-mode FA \
        --freeze-patch-embedding \
        --sbora-base-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 43

# KITTI dataset SBoRA fine tuning with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29412 \
        --use_env fine_tune_sbora.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/KITTI \
        --image-dataset KITTI \
        --selected-multimodality image \
        --checkpoint-dir fine_tune_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_image_fine_tune_KITTI_bs128_ddp \
        --log-file fine_tune_checkpoints/train_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_image_fine_tune_KITTI_bs128_ddp.log \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --sbora-rank 128 \
        --sbora-alpha 128 \
        --sbora-dropout 0.0 \
        --sbora-mode FA \
        --freeze-patch-embedding \
        --sbora-base-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 9

# MNIST dataset SBoRA fine tuning with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29412 \
        --use_env fine_tune_sbora.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/MNIST \
        --image-dataset MNIST \
        --selected-multimodality image \
        --checkpoint-dir fine_tune_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_image_fine_tune_MNIST_bs128_ddp \
        --log-file fine_tune_checkpoints/train_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_image_fine_tune_MNIST_bs128_ddp.log \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --sbora-rank 128 \
        --sbora-alpha 128 \
        --sbora-dropout 0.0 \
        --sbora-mode FA \
        --freeze-patch-embedding \
        --sbora-base-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 10

# RESISC45 dataset SBoRA fine tuning with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29412 \
        --use_env fine_tune_sbora.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/RESISC45 \
        --image-dataset RESISC45 \
        --selected-multimodality image \
        --checkpoint-dir fine_tune_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_image_fine_tune_RESISC45_bs128_ddp \
        --log-file fine_tune_checkpoints/train_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_image_fine_tune_RESISC45_bs128_ddp.log \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --sbora-rank 128 \
        --sbora-alpha 128 \
        --sbora-dropout 0.0 \
        --sbora-mode FA \
        --freeze-patch-embedding \
        --sbora-base-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 45

# SUN397 dataset SBoRA fine tuning with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29412 \
        --use_env fine_tune_sbora.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/SUN397 \
        --image-dataset SUN397 \
        --selected-multimodality image \
        --checkpoint-dir fine_tune_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_image_fine_tune_SUN397_bs128_ddp \
        --log-file fine_tune_checkpoints/train_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_image_fine_tune_SUN397_bs128_ddp.log \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --sbora-rank 128 \
        --sbora-alpha 128 \
        --sbora-dropout 0.0 \
        --sbora-mode FA \
        --freeze-patch-embedding \
        --sbora-base-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 397

# SVHN dataset SBoRA fine tuning with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29412 \
        --use_env fine_tune_sbora.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/SVHN \
        --image-dataset SVHN \
        --selected-multimodality image \
        --checkpoint-dir fine_tune_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_image_fine_tune_SVHN_bs128_ddp \
        --log-file fine_tune_checkpoints/train_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_image_fine_tune_SVHN_bs128_ddp.log \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --sbora-rank 128 \
        --sbora-alpha 128 \
        --sbora-dropout 0.0 \
        --sbora-mode FA \
        --freeze-patch-embedding \
        --sbora-base-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 10

#######################################################################################################################
# SBoRA Fine Tuning Script for audio modality #
# VGGSound dataset SBoRA fine tuning with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29712 \
        --use_env fine_tune_sbora.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality audio \
        --audio-data-path /data_ssd/DATA/VGGSound \
        --audio-dataset vggsound \
        --selected-multimodality audio \
        --checkpoint-dir fine_tune_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_audio_fine_tune_vggsound_bs128_ddp \
        --log-file fine_tune_checkpoints/train_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_audio_fine_tune_vggsound_bs128_ddp.log \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --sbora-rank 128 \
        --sbora-alpha 128 \
        --sbora-dropout 0.0 \
        --sbora-mode FA \
        --freeze-patch-embedding \
        --sbora-base-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 309

# EpicSound dataset SBoRA fine tuning with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29712 \
        --use_env fine_tune_sbora.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality audio \
        --audio-data-path /data1/steven/DATA/EPIC-Sounds-wav \
        --audio-dataset epicsound \
        --selected-multimodality audio \
        --checkpoint-dir fine_tune_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_audio_fine_tune_epicsound_bs128_ddp \
        --log-file fine_tune_checkpoints/train_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_audio_fine_tune_epicsound_bs128_ddp.log \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --sbora-rank 128 \
        --sbora-alpha 128 \
        --sbora-dropout 0.0 \
        --sbora-mode FA \
        --freeze-patch-embedding \
        --sbora-base-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 44

# SpeechCommand dataset SBoRA fine tuning with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29712 \
        --use_env fine_tune_sbora.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality audio \
        --audio-data-path /data_ssd/DATA/Speech-Command-V2 \
        --audio-dataset speechcommand \
        --selected-multimodality audio \
        --checkpoint-dir fine_tune_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_audio_fine_tune_speechcommand_bs128_ddp \
        --log-file fine_tune_checkpoints/train_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_audio_fine_tune_speechcommand_bs128_ddp.log \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --sbora-rank 128 \
        --sbora-alpha 128 \
        --sbora-dropout 0.0 \
        --sbora-mode FA \
        --freeze-patch-embedding \
        --sbora-base-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 35


# Nsynth dataset SBoRA fine tuning with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29712 \
        --use_env fine_tune_sbora.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality audio \
        --audio-data-path /data_ssd/DATA/nsynth_data \
        --audio-dataset nsynth \
        --selected-multimodality audio \
        --checkpoint-dir fine_tune_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_audio_fine_tune_nsynth_bs128_ddp \
        --log-file fine_tune_checkpoints/train_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_audio_fine_tune_nsynth_bs128_ddp.log \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --sbora-rank 128 \
        --sbora-alpha 128 \
        --sbora-dropout 0.0 \
        --sbora-mode FA \
        --freeze-patch-embedding \
        --sbora-base-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 11

########################################################################################################################
# SBoRA Fine Tuning for text modality #
# AGNews dataset SBoRA fine tuning with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=23434 \
        --use_env fine_tune_sbora.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality text \
        --text-data-path /data_ssd/DATA/AG_NEWS \
        --text-dataset agnews \
        --selected-multimodality text \
        --checkpoint-dir fine_tune_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_audio_fine_tune_agnews_bs128_ddp \
        --log-file fine_tune_checkpoints/train_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_audio_fine_tune_agnews_bs128_ddp.log \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --sbora-rank 128 \
        --sbora-alpha 128 \
        --sbora-dropout 0.0 \
        --sbora-mode FA \
        --freeze-patch-embedding \
        --sbora-base-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 4

# NewsGroups20 dataset SBoRA fine tuning with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities.
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=23434 \
        --use_env fine_tune_sbora.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality text \
        --text-data-path /data_ssd/DATA/20NEWSGROUPS \
        --text-dataset newsgroups20 \
        --selected-multimodality text \
        --checkpoint-dir fine_tune_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_audio_fine_tune_newsgroups20_bs128_ddp \
        --log-file fine_tune_checkpoints/train_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_audio_fine_tune_newsgroups20_bs128_ddp.log \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --sbora-rank 128 \
        --sbora-alpha 128 \
        --sbora-dropout 0.0 \
        --sbora-mode FA \
        --freeze-patch-embedding \
        --sbora-base-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 20

# IMDB dataset SBoRA fine tuning with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities.
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=23434 \
        --use_env fine_tune_sbora.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality text \
        --text-data-path /data_ssd/DATA/IMDB \
        --text-dataset imdb \
        --selected-multimodality text \
        --checkpoint-dir fine_tune_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_audio_fine_tune_imdb_bs128_ddp \
        --log-file fine_tune_checkpoints/train_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_audio_fine_tune_imdb_bs128_ddp.log \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --sbora-rank 128 \
        --sbora-alpha 128 \
        --sbora-dropout 0.0 \
        --sbora-mode FA \
        --freeze-patch-embedding \
        --sbora-base-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 2


# CARER dataset SBoRA fine tuning with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities.
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=23434 \
        --use_env fine_tune_sbora.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality text \
        --text-data-path /data_ssd/DATA/CARER \
        --text-dataset carer \
        --selected-multimodality text \
        --checkpoint-dir fine_tune_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_audio_fine_tune_carer_bs128_ddp \
        --log-file fine_tune_checkpoints/train_pretrained_Omni-C_image_audio_text_sbora_rank128_alpha128_audio_fine_tune_carer_bs128_ddp.log \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --sbora-rank 128 \
        --sbora-alpha 128 \
        --sbora-dropout 0.0 \
        --sbora-mode FA \
        --freeze-patch-embedding \
        --sbora-base-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 6