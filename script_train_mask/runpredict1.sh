python -m torch.distributed.launch --nproc_per_node 1 main.py \
    --data_predict data/ \
    --predict \
    --batch-size 8 \
    --cfg configs/swin_large_patch4_window12_384.yaml \
    --data-path dataset/mask_kfolddata/fold1 \
    --resume output_mask_kfoldshuffle/fold1_best/swin_large_patch4_window12_384/default/ckpt_epoch_12.pth \
    --class_type mask \
    --version_model fold1_e12 \
    --predict_xs \