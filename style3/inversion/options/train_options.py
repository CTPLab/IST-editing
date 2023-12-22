from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class TrainOptions:
    """ Defines all training arguments. """

    """ General Args """
    # Path to experiment output directory
    exp_dir: Path = ''
    # Which decoder to use
    decoder: str = 'style2'
    # Type of dataset/experiment to run
    dataset_type: str = 'CosMx'
    # Type of datasplit for calc eig
    data_splt: str = None
    # Type of gene formality
    gene_type: str = 'tabular'
    # The amount of gene expr
    gene_num: int = 0
    # Whether to train with gene feat
    gene_use: bool = False
    # Whether to train with gene local
    gene_local: bool = False
    # The amount of gene expr
    kernel_size: int = 3
    # Whether use config-r for stylegan3
    use_cfgr: bool = False
    # Which encoder to use
    encoder_type: str = 'ResNetBackboneEncoder'
    # Number of input image channels to the ReStyl encoder
    input_nc: int = 3
    # Output resolution of the generator
    output_size: int = 128
    # Number of forward passes per batch during training
    n_iters_per_batch: int = 1

    """ Dataset args """
    # Batch size for training
    batch_size: int = 8
    # Batch size of testing/validation
    test_batch_size: int = 0
    # Number of workers for train dataloader
    workers: int = 0
    # Number of works for test dataloader
    test_workers: int = 0

    """ Optimizer args """
    # Optimizer learning rate
    learning_rate: float = 0.0001
    # Which optimizer to use
    optim_name: str = 'ranger'
    # Whether to train the decoder during training
    train_decoder: bool = False
    # Whether to add average latent vector to generate codes from encoder
    start_from_latent_avg: bool = False

    """ Loss args """
    # LPIPS loss multiplier factor
    lpips_lambda: float = 0
    # ID loss multiplier factor
    id_lambda: float = 0
    # L2 loss multiplier factor
    l2_lambda: float = 0
    # W-norm loss multiplier factor
    w_norm_lambda: float = 0
    # Moco feature loss multiplier factor
    moco_lambda: float = 0

    """ Checkpoint args """
    # Path to StyleGAN model weights
    stylegan_weights: Path = None
    # Path to ReStyle model checkpoint for resuming training from
    checkpoint_path: Optional[Path] = None

    """ Logging args """
    # Maximum number of training steps
    max_steps: int = 500000
    # Interval for logging train images during training
    image_interval: int = 100
    # Interval for logging metrics to tensorboard
    board_interval: int = 50
    # Validation interval
    val_interval: int = 1000
    # Model checkpoint interval
    save_interval: Optional[int] = None
    # Number of batches to run validation on. If None, run on all batches
    max_val_batches: Optional[int] = None

    device: Optional[str] = None

    w_discriminator_lambda: float = 0
    w_discriminator_lr: float = 2e-5
    r1: float = 10
    d_reg_every: int = 16

    def update(self, new_opts):
        for key, value in new_opts.items():
            setattr(self, key, value)
