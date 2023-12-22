from pathlib import Path

dataset_paths = {
    'celeba_train': Path(''),
    'celeba_test': Path(''),

    'ffhq': Path(''),
    'ffhq_unaligned': Path(''),

    'ham10k': Path('Data/non_IID/preproc/ham10k_tiny/images'),
    'rxrx19b': Path('Data/non_IID/preproc/rxrx19b_cell/images')
}

model_paths = {
    'moco': Path('style3/pretrained_models/moco_v2_800ep_pretrain.pt')
}

styleclip_directions = {
    "ffhq": {
        'delta_i_c': Path('editing/styleclip_global_directions/sg3-r-ffhq-1024/delta_i_c.npy'),
        's_statistics': Path('editing/styleclip_global_directions/sg3-r-ffhq-1024/s_stats'),
    },
    'templates': Path('editing/styleclip_global_directions/templates.txt')
}

interfacegan_aligned_edit_paths = {
    'age': Path('editing/interfacegan/boundaries/ffhq/age_boundary.npy'),
    'smile': Path('editing/interfacegan/boundaries/ffhq/Smiling_boundary.npy'),
    'pose': Path('editing/interfacegan/boundaries/ffhq/pose_boundary.npy'),
    'Male': Path('editing/interfacegan/boundaries/ffhq/Male_boundary.npy'),
}

interfacegan_unaligned_edit_paths = {
    'age': Path('editing/interfacegan/boundaries/ffhqu/age_boundary.npy'),
    'smile': Path('editing/interfacegan/boundaries/ffhqu/Smiling_boundary.npy'),
    'pose': Path('editing/interfacegan/boundaries/ffhqu/pose_boundary.npy'),
    'Male': Path('editing/interfacegan/boundaries/ffhqu/Male_boundary.npy'),
}
