python -m torch.distributed.launch --nproc_per_node 1 main.py \
    --data_predict data/ \
    --predict \
    --batch-size 8 \
    --cfg configs/swin_large_patch4_window12_384.yaml \
    --data-path dataset/mask_kfolddata/fold5 \
    --resume output_mask_kfoldshuffle/fold5_best/swin_large_patch4_window12_384/default/ckpt_epoch_14.pth \
    --class_type mask \
    --version_model fold5_e14 \
    --predict_xs \