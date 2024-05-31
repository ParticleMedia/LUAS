# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.



"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from training_scripts.utils.model_utils import *
sys.path.append(str(Path(__file__).resolve().parent.parent))


def train(model,
          train_dataloader,
          train_sampler,
          eval_dataloader,
          tokenizer,
          optimizer,
          lr_scheduler,
          gradient_accumulation_steps,
          train_config,
          fsdp_config=None,
          local_rank=None,
          rank=None,
          first_step=0):
    """
    Trains the model on the given dataloader
    
    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        train_sampler: The sampler for the training data, used to shuffle between epoches
        eval_dataloader: The dataloader containing the validation data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons
    
    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss = []
    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    # import pdb; pdb.set_trace()
    wandb_writer = WanDBWriter(project=train_config.wandb_project, name=train_config.wandb_name, rank=rank)

    accu_step = first_step
    for epoch in range(train_config.num_epochs):
        train_sampler.set_epoch(epoch)
        epoch_start_time = time.perf_counter()

        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            for step, batch in enumerate(tqdm(train_dataloader, colour="blue", desc=f"Training Epoch{epoch}")):
                accu_step += 1
                for key in batch.keys():
                    if train_config.enable_fsdp:
                        batch[key] = batch[key].to(local_rank)
                    else:
                        batch[key] = batch[key].to('cuda:0')
                loss = model(**batch).loss
                loss = loss / gradient_accumulation_steps

                # 如果 loss 出现nan，放弃这一步更新
                if torch.isnan(loss).any():
                    continue

                total_loss += loss.detach().float()
                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        if hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(train_config.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                model.parameters(), train_config.max_grad_norm,
                            )
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()

                if train_config.enable_fsdp:
                    if rank == 0:
                        print(f"\n step {step} is completed and loss is {loss.detach().float()}")
                else:
                    print(f"\n step {step} is completed and loss is {loss.detach().float()}")

                if accu_step % train_config.check_point_steps == 0 and not torch.isnan(loss).any():
                    save_model(model, train_config, fsdp_config, rank, optimizer, accu_step=accu_step)
                    if train_config.run_validation:
                        eval_ppl, eval_epoch_loss = evaluation(model, train_config, eval_dataloader, rank, tokenizer)
                        wandb_writer.log(rank, {
                            'step': accu_step // gradient_accumulation_steps,
                            'valid_loss': eval_epoch_loss
                        })

                if accu_step % train_config.evaluation_steps == 0:
                    eval_ppl, eval_epoch_loss = evaluation(model, train_config, eval_dataloader, rank, tokenizer)
                    val_loss.append(best_val_loss)
                    val_prep.append(eval_ppl)
                    wandb_writer.log(rank, {
                        'step': accu_step // gradient_accumulation_steps,
                        'valid_loss': eval_epoch_loss
                    })

                wandb_writer.log(rank, {
                    'step': accu_step // gradient_accumulation_steps,
                    'step_loss': loss.detach().float(),
                    'learning_rate': lr_scheduler.get_lr()[0]
                })

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss / world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)

        if train_config.enable_fsdp:
            if rank == 0:
                print(f"Max CUDA memory allocated was {memtrace.peak} GB")
                print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
                print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
                print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
                print(
                    f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        else:
            print(f"Max CUDA memory allocated was {memtrace.peak} GB")
            print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
            print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            print(
                f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")

        save_model(model, train_config, fsdp_config, rank, None, epoch=epoch, accu_step=accu_step)

        if train_config.run_validation:
            eval_ppl, eval_epoch_loss = evaluation(model, train_config, eval_dataloader, rank, tokenizer)
            checkpoint_start_time = time.perf_counter()
            if train_config.use_peft and train_config.save_model and eval_epoch_loss < best_val_loss:
                save_model(model, train_config, fsdp_config, rank, optimizer, sub_dir='best_model')

            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.enable_fsdp:
                    if rank == 0:
                        print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                else:
                    print(f"best eval loss on epoch {epoch} is {best_val_loss}")

            val_loss.append(best_val_loss)
            val_prep.append(eval_ppl)
            wandb_writer.log(rank, {
                'step': accu_step // gradient_accumulation_steps,
                'valid_loss': eval_epoch_loss
            })

        if train_config.enable_fsdp:
            if rank == 0:
                print(
                    f"Epoch {epoch + 1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epcoh time {epoch_end_time}s")
        else:
            print(
                f"Epoch {epoch + 1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epcoh time {epoch_end_time}s")

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times) / len(checkpoint_times)
    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep) / len(val_prep)
        avg_eval_loss = sum(val_loss) / len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time

    # saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft:
        save_dir = train_config.output_dir + '/model_final'
        save_train_params(save_dir, train_config, fsdp_config, rank)

    return results


def train_partition(model,
                    train_dataloader,
                    train_sampler,
                    eval_dataloader,
                    tokenizer,
                    optimizer,
                    lr_scheduler,
                    gradient_accumulation_steps,
                    train_config,
                    fsdp_config=None,
                    local_rank=None,
                    rank=None,
                    first_step=0):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        train_sampler: The sampler for the training data, used to shuffle between epoches
        eval_dataloader: The dataloader containing the validation data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    # import pdb; pdb.set_trace()
    wandb_writer = WanDBWriter(project=train_config.wandb_project, name=train_config.wandb_name, rank=rank)

    accu_step = first_step
    train_sampler.set_epoch(first_step)
    epoch_start_time = time.perf_counter()

    with MemoryTrace() as memtrace:  # track the memory usage
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader, colour="blue", desc=f"Training")):
            accu_step += 1

            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to('cuda:0')
            loss = model(**batch).loss
            loss = loss / gradient_accumulation_steps

            total_loss += loss.detach().float()
            if train_config.use_fp16:
                # if fp16 is enabled, use gradient scaler to handle gradient update
                scaler.scale(loss).backward()
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)

                    # 如果 loss 出现nan，放弃这一步更新，梯度直接置零，不更新
                    if torch.isnan(loss).any():
                        optimizer.zero_grad()

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # regular backpropagation when fp16 is not used
                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    if hasattr(model, "clip_grad_norm_"):
                        # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                        model.clip_grad_norm_(train_config.max_grad_norm)
                    else:
                        # Revert to normal clipping otherwise, handling Apex or full precision
                        nn.utils.clip_grad_norm_(
                            model.parameters(), train_config.max_grad_norm,
                        )

                    # 如果 loss 出现nan，放弃这一步更新，梯度直接置零，不更新
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        optimizer.zero_grad()

                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

            wandb_writer.log(rank, {
                'step': accu_step // gradient_accumulation_steps,
                'step_loss': loss.detach().float(),
                'learning_rate': lr_scheduler.get_lr()[0]
            })

            # 改为先做 Evaluation
            # stream based pre-train eval 使用 step，不使用 accu_step
            if (accu_step % train_config.check_point_steps == 0
                    or accu_step % train_config.evaluation_steps == 0
                    or (step + 1) == len(train_dataloader)
            ):
                if accu_step % train_config.check_point_steps == 0 and not torch.isnan(loss).any():
                    save_model(model, train_config, fsdp_config, rank, None, accu_step=accu_step)

                # validation
                eval_ppl, eval_epoch_loss = evaluation(model, train_config, eval_dataloader, rank, tokenizer)
                wandb_writer.log(rank, {
                    'step': accu_step // gradient_accumulation_steps,
                    'valid_loss': eval_epoch_loss
                })

            if not train_config.enable_fsdp or rank == 0:
                print(f"\n step {step} is completed and loss is {loss.detach().float()}")

    epoch_end_time = time.perf_counter() - epoch_start_time
    # Reducing total_loss across all devices if there's more than one CUDA device
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    train_epoch_loss = total_loss / len(train_dataloader)
    if train_config.enable_fsdp:
        train_epoch_loss = train_epoch_loss / world_size
    train_perplexity = torch.exp(train_epoch_loss)

    if not train_config.enable_fsdp or rank == 0:
        print(f"Max CUDA memory allocated was {memtrace.peak} GB")
        print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
        print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
        print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
        print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        print(
            f"step {first_step} to {accu_step}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epcoh time {epoch_end_time}s")

    print(f'\ntraining rank {rank} is done.\n', flush=True)
    # save_model(model, train_config, fsdp_config, rank, None, accu_step=accu_step)

    return accu_step



