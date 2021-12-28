python3 -m torch.distributed.launch --nproc_per_node 1 main.py \
--cfg configs/swin_large_patch4_window12_384.yaml \
--data-path dataset/mask_kfolddata/fold2 \
--batch-size 8 \
--output output_mask_kfoldshuffle/fold2_best \
--resume swin_large_patch4_window12_384_22kto1k.pth \
--use-checkpoint \
--class_type mask \
--opts TRAIN.EPOCHS 20 TRAIN.WARMUP_EPOCHS 5 MODEL.DROP_PATH_RATE 0.1 AUG.MIXUP 0.0 AUG.CUTMIX 0.0 TRAIN.BASE_LR 5e-4 TRAIN.WARMUP_LR 5e-5 TRAIN.MIN_LR 5e-6 \

