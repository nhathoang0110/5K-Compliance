python -m torch.distributed.launch --nproc_per_node 1 main.py \
    --data_predict data/ \
    --predict \
    --batch-size 8 \
    --cfg configs/swin_large_patch4_window12_384.yaml \
    --data-path dataset/mask_kfolddata/fold2 \
    --resume output_mask_kfoldshuffle/fold2_best/swin_large_patch4_window12_384/default/ckpt_epoch_13.pth \
    --class_type mask \
    --version_model fold2_e13 \
    --predict_xs \