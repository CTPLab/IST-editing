python -m infer_brn \
    --seed=0 \
    --data_name=609882 \
    --analysis=GAN --task=$TASK \
    --gene_num=500 \
    --input_nc=1 \
    --n_eval=16 --n_iter=$N_ITER \
    --cell_label=kmeans_11_clusters \
    --data_path=Data/MERFISH/ \
    --ckpt_path=Data/MERFISH/exp_609882_True_0.5_slide_ID_numeric_3_512_crop_0/checkpoint \
    --save_path=Experiment_inf/MERFISH/