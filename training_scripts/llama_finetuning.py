# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import pathlib
import time

import fire

# Unused imports removed
from training_scripts.utils import fsdp_auto_wrap_policy
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    default_data_collator,
    BitsAndBytesConfig
)
import torch.distributed as dist
# Unused imports removed
from training_scripts.utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    print_model_size,
    get_policies  
)

from training_scripts.utils.dataset_utils import get_preprocessed_dataset

from training_scripts.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from peft import (
    PeftModel, PeftConfig,
    get_peft_model, TaskType, prepare_model_for_int8_training
)

import configs
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.utils.data import DistributedSampler
from training_scripts import policies
from training_scripts.policies import AnyPrecisionAdamW
from training_scripts.configs import fsdp_config, train_config
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from training_scripts.model_checkpointing import load_optimizer_checkpoint


def main(**kwargs):

    # Update the configuration for the training and sharding process
    update_config((train_config, fsdp_config), **kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1.

    if torch.distributed.is_initialized():
        torch.cuda.set_device(rank)
        setup_environ_flags(rank)
    
    # Calculate gradient accumulation steps
    gradient_accumulation_steps = train_config.train_batch_size // train_config.micro_batch_size

    model_name = train_config.model_name
    # gpu3 cpu memory is not enough, lazy loading with 20s after
    if train_config.pre_train_model_path and pathlib.Path(train_config.pre_train_model_path).exists():
        model_name = train_config.pre_train_model_path

    print(f'{"x" * 20}    {model_name}    {"x" * 20}')

    # Load the pre-trained model and setup its configuration
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True if train_config.quantization else None,
        device_map="auto" if train_config.quantization else None,
    )
    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            # model = BetterTransformer.transform(model) 
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)
    
    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)
        
    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )

    if train_config.peft_model:
        model = PeftModel.from_pretrained(model, train_config.peft_model)
        for name, param in model.named_parameters():
            if 'lora' in name or 'Lora' in name:
                param.requires_grad = True
        model.print_trainable_parameters()

    elif train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            
            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
   
        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            policies.apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")

    dataset_config = generate_dataset_config(train_config, kwargs)
    if train_config.dataset_dir != '':
        dataset_config.dataset_dir = train_config.dataset_dir
    print(dataset_config.dataset, dataset_config.dataset_dir)
    # import pdb; pdb.set_trace()
    
     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )
    
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    if not train_config.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")

    train_sampler = None
    val_sampler = None
    if train_config.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset_train,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
        )
        if train_config.run_validation:
            val_sampler = DistributedSampler(
                dataset_val,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
            )
        
    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.micro_batch_size,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.valid_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            collate_fn=default_data_collator,
        )
        
    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=0.0,
            eps=1e-8,
            betas=(0.9, 0.999),
        )

    if train_config.checkpoint_optimizer:
        from pathlib import Path
        path = Path(train_config.checkpoint_optimizer)
        if path.exists():
            sharded_osd = load_optimizer_checkpoint(model, Path(train_config.checkpoint_optimizer), rank)
            optimizer.load_state_dict(sharded_osd)
            del sharded_osd
            torch.cuda.empty_cache()

    from transformers import get_scheduler
    num_training_steps = train_config.num_epochs * len(train_dataloader)
    if rank == 0:
        print(f'num training steps = {num_training_steps}')
        print(f'num eval steps = {len(eval_dataloader)}')
        print(f'num data batches = {len(train_dataloader)}')

    from transformers import get_scheduler, SchedulerType
    total_steps = len(train_dataloader) * train_config.num_epochs
    scheduler = get_scheduler(
        SchedulerType.COSINE,
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * 0.03),
        num_training_steps=total_steps,
    )

    # Start the training process
    results = train(
        model,
        train_dataloader,
        train_sampler,
        eval_dataloader, 
        tokenizer,
        optimizer,
        scheduler,
        gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    fire.Fire(main)
