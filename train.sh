python train_style2_inf.py \
    Data/Xenium_mouse/GAN/crop \
    --data=Xenium_mouse \
    --gene=379 \
    --batch=8 \
    --iter=800000 \
    --size=128 \
    --channel=-1 \
    --kernel_size=3 \
    --gene_use \
    --cell_label=kmeans_10_clusters \
    --train_sub=all \
    --split_scheme=slide_ID_numeric \
    --stain=dapi \
    --img_chn=1 --latent=512 --mixing=0.5 \
    --check_save=Data/Xenium_mouse/GAN/dapi