python -m torch.distributed.launch --nproc_per_node 1 main.py \
    --data_predict data/ \
    --predict \
    --batch-size 8 \
    --cfg configs/swin_large_patch4_window12_384.yaml \
    --data-path dataset/mask_kfolddata/fold4 \
    --resume output_mask_kfoldshuffle/fold4_best/swin_large_patch4_window12_384/default/ckpt_epoch_9.pth \
    --class_type mask \
    --version_model fold4_e9 \
    --predict_xs \