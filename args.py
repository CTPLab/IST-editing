import random
from pathlib import Path
from argparse import ArgumentParser


def _get_pths(args):
    save_data = f'{args.analysis}_{args.task}'
    save_info = f'{args.data_splt}_{args.proc}'
    if args.decoder == 'style2':
        save_info += '_style2'
    elif args.decoder == 'style3':
        if 'stylegan3-t' in args.ckpt_path.name:
            save_info += '_style3t'
        elif 'stylegan3-r' in args.ckpt_path.name:
            save_info += '_style3r'
    if args.analysis != 'RAW':
        save_info += f'_{args.ckpt_path.parent.name}_{args.seed}'
    args.save_path = args.save_path / save_data / save_info
    args.save_path.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Global seed (for weight initialization, data sampling, etc.). '
                             'If not specified it will be randomized (and printed on the log)')

    parser.add_argument('--task',
                        type=str,
                        help='Calculate the stats or output image plots, video demos.')

    parser.add_argument('--decoder',
                        type=str,
                        default='style2')

    parser.add_argument('--encoder',
                        type=str,
                        default='psp')

    parser.add_argument('--analysis',
                        type=str,
                        help='The object to be analyzed such as raw metadata, GAN (Inversion) output')

    parser.add_argument('--proc', type=str, choices=('crop', 'tile'),
                        default='crop',
                        help='processing method (center-crop or tile with fixed gap)')
    parser.add_argument('--gene_num', type=int)
    parser.add_argument('--gene_spa', action='store_true')
    parser.add_argument('--gene_use', action='store_true')
    parser.add_argument('--gene_name', type=str, default='')
    parser.add_argument('--cell_label', type=str, default='')
    parser.add_argument('--subtype', action='store_true')
    parser.add_argument('--fov', action='store_true')
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--output_size', type=int, default=128)
    parser.add_argument('--use_cfgr', action='store_true')

    parser.add_argument('--data_name',
                        type=str,
                        choices=('CosMx', 'Xenium', 'Xenium_brain', 'Visium', 'Xenium_mouse'),
                        help='Name of ST platforms')
    parser.add_argument('--data_splt',
                        type=str,
                        default=None,
                        help='Name of data splitting,'
                             'slide_ID_numeric means split data between normal and tumor slide')
    parser.add_argument('--data_path',
                        type=Path,
                        help='path to the data root.')

    parser.add_argument('--ckpt_path',
                        type=Path,
                        help='Path to the model checkpoint')

    parser.add_argument('--save_path',
                        type=Path,
                        help='Path to saved output')

    parser.add_argument('--n_iter',
                        type=int,
                        default=None,
                        help='Training iterations of the checkpoint')
    parser.add_argument('--n_eval',
                        type=int,
                        default=8,
                        help='Batch size')
    parser.add_argument('--n_work',
                        type=int,
                        default=8,
                        help='The amount of data loader workers')
    parser.add_argument('--input_nc',
                        type=int,
                        default=1,
                        help='The amount of data loader workers')

    args = parser.parse_args()

    assert args.encoder == 'psp'

    if args.analysis == 'GAN':
        assert not args.gene_spa

    if args.seed is None:
        args.seed = random.randint(0, 10 ** 9)

    args.size_bat = args.n_eval
    if args.data_name == 'CosMx':
       assert args.input_nc == 2
    elif args.data_name == 'Xenium_breast':
        assert args.input_nc == 3
    elif args.data_name == 'Xenium_mouse':
        assert args.input_nc in (1, 3)
        args.stain = 'he' if args.input_nc == 3 else 'dapi'

    _get_pths(args)

    return args
