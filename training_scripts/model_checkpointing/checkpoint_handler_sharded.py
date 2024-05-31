# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from pathlib import Path
from datetime import datetime
import torch
import time

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)

from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    save_state_dict,
    load_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)

from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed._shard.checkpoint as dist_cp
import torch.distributed as dist


# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


def save_model_and_optim_sharded(model, rank, save_dir, optim=None, accu_step=None):
    """save model and optimizer via sharded_state_dict to save_dir"""
    if rank == 0:
        print(f"Saving model to {save_dir}")

    distributed_writer = dist_cp.FileSystemWriter(save_dir)
    t0 = time.perf_counter()

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):

        state_dict = {"model": model.state_dict()}
        if optim is not None:
            torch.save(optim, f'{save_dir}/optim.{rank}')
            # state_dict["optim"] = FSDP.optim_state_dict(model, optim)

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=distributed_writer,
            planner=DefaultSavePlanner(),
        )
    dist.barrier()
    t1 = time.perf_counter()
    if rank == 0:
        print(f"Sharded state checkpoint saved to {save_dir}")
        print(
            f"Checkpoint Time = {t1 - t0:.4f}\n"
        )


def load_sharded_model_single_gpu(model, model_path):
    state_dict = {"model": model.state_dict()}
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=FileSystemReader(model_path),
        no_dist=True,
    )

    model.load_state_dict(state_dict["model"])
    print(f"Sharded state checkpoint loaded from {model_path}")
    return model


if __name__ == '__main__':
    file_path = './'
    reader = FileSystemReader(file_path)
    meta = reader.read_metadata()
    print(meta.state_dict_metadata.keys())
