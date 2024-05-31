# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class train_config:
    model_name: str="PATH/to/LLAMA/7B"
    pre_train_model_path: str=""
    enable_fsdp: bool= False
    run_validation: bool=True
    train_batch_size: int=1
    num_epochs: int=3
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    valid_batch_size: int=1
    dataset = "samsum_dataset"
    micro_batch_size: int=1
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    check_point_steps: int = 1
    evaluation_steps: int = 100
    wandb_name: str = 'default'
    wandb_project: str = 'default'
    peft_model: str = ''
    max_grad_norm: float = 1.0
    dataset_dir: str = ''
    max_ckpt_num: int = 5

    # restart pre-training config
    checkpoint_data_file: str = ''
    checkpoint_optimizer: str = ''

