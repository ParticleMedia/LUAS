import os
import sys
from typing import List
import yaml
import time

import fire
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm

import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from training_scripts import model_checkpointing
from transformers import LlamaForCausalLM, LlamaTokenizer
from pkg_resources import packaging
from .memory_utils import MemoryTrace
from torch import nn
import torch.cuda.nccl as nccl
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from training_scripts.policies import bfSixteen, fpSixteen, bfSixteen_mixed, get_llama_wrapper
import wandb


class WanDBWriter:
    def __init__(self, project, name, rank):
        if rank == 0:
            wandb.init(
                project=project,
                entity="canon",
                name=name
            )
            wandb.define_metric('step')
            wandb.define_metric('learning_rate', step_metric='step')
            wandb.define_metric('step_loss', step_metric='step')
            wandb.define_metric('valid_loss', step_metric='step')

    def log(self, rank, info):
        if rank == 0:
            wandb.log(info)


def remove_exist_saved_models(train_config):
    model_dirs = []
    for sub_dir in os.listdir(train_config.output_dir):
        if sub_dir.startswith('step_'):
            model_dirs.append(train_config.output_dir + '/' + sub_dir)
    model_dirs.sort()
    if len(model_dirs) <= train_config.max_ckpt_num:
        return
    remove_dirs = model_dirs[0:-train_config.max_ckpt_num]
    for remove_dir in remove_dirs:
        try:
            os.system(f'rm -r {remove_dir}')
        except:
            print(f'delete {remove_dir} failed')


def save_model(model, train_config, fsdp_config, rank, optimizer=None, epoch=-1, accu_step=-1, sub_dir=''):

    if epoch != -1:
        sub_dir = f'epoch_{str(1000 + epoch)[1:]}'
    elif accu_step != -1:
        if optimizer:
            sub_dir = f'optimizer_step_{str(1000000 + accu_step)[1:]}'
        else:
            sub_dir = f'step_{str(1000000 + accu_step)[1:]}'
    elif not sub_dir:
        return

    save_dir = train_config.output_dir + '/' + sub_dir
    if train_config.enable_fsdp:
        dist.barrier()
    if train_config.use_peft:
        if not train_config.enable_fsdp or rank == 0:
            print(f"we are about to save the PEFT modules")
        model.save_pretrained(save_dir)
        if not train_config.enable_fsdp or rank == 0:
            print(f"PEFT modules are saved in {save_dir} directory")

    else:
        if fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
            model_checkpointing.save_model_checkpoint(
                model, optimizer, rank, save_dir, train_config, epoch=epoch
            )
        elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
            print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
            model_checkpointing.save_model_and_optim_sharded(model, rank, save_dir, None, accu_step)

        if optimizer:
            model_checkpointing.save_optimizer_checkpoint(
                model, optimizer, rank, save_dir, accu_step=accu_step
            )
            print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
    if train_config.enable_fsdp:
        dist.barrier()

    remove_exist_saved_models(train_config)

    return save_dir


def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    eval_preds = []
    eval_loss = 0.0  # Initialize evaluation loss
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc="evaluating Epoch")):
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            eval_preds.extend(
                tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            )

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss / world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank == 0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")
    model.train()
    return eval_ppl, eval_epoch_loss


def freeze_transformer_layers(model, num_layer):
    for i, layer in enumerate(model.model.layers):
        if i < num_layer:
            for param in layer.parameters():
                param.requires_grad = False


def check_frozen_layers_peft_model(model):
    for i, layer in enumerate(model.base_model.model.model.layers):
        for name, param in layer.named_parameters():
            print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes


def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""

    verify_bfloat_support = (
            torch.version.cuda
            and torch.cuda.is_bf16_supported()
            and packaging.version.parse(torch.version.cuda).release >= (11, 0)
            and dist.is_nccl_available()
            and nccl.version() >= (2, 10)
    )

    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy


def save_train_params(save_dir, train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    # If the directory does not exist, create it
    os.makedirs(save_dir, exist_ok=True)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir, 'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank == 0:
            print(f"training params are saved in {file_name}")



def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"


# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2 ** 20)
