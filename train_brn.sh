python train_style2_brn.py \
    Data/MERFISH/ \
    --data=609882 \
    --gene=500 \
    --batch=8 \
    --iter=800000 \
    --size=128 \
    --channel=-1 \
    --kernel_size=3 \
    --gene_use \
    --cell_label=kmeans_10_clusters \
    --split_scheme=slide_ID_numeric \
    --stain=DAPI \
    --img_chn=1 --latent=512 --mixing=0.5 \
    --check_save=Data/MERFISH/exp_609882