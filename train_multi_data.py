import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):

    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    wandb_logger = None
    if cfg.using_wandb:
        import wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                sync_tensorboard=True,
                resume=cfg.wandb_resume,
                name=run_name,
                notes=cfg.notes) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")

    # === Multi-dataset: create N dataloaders ===
    num_dataset = len(cfg.rec)
    assert num_dataset == len(cfg.num_classes) == len(cfg.num_image) == len(cfg.batch_size) == len(cfg.loss_w), \
        "cfg.rec, cfg.num_classes, cfg.num_image, cfg.batch_size, cfg.loss_w must have the same length"

    train_loader_dict = {}
    for i in range(num_dataset):
        train_loader = get_dataloader(
            cfg.rec[i],
            local_rank,
            cfg.batch_size[i],
            cfg.dali,
            cfg.dali_aug,
            cfg.seed,
            cfg.num_workers
        )
        train_loader_dict[i] = train_loader

    # === Backbone (shared) ===
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)

    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()

    # === N PartialFC_V2 modules ===
    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    module_partial_fc_dict = {}
    temp_params = [{"params": backbone.parameters()}]

    for i in range(num_dataset):
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes[i],
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        module_partial_fc_dict[i] = module_partial_fc
        temp_params.append({"params": module_partial_fc.parameters()})

    # === Single optimizer for all param groups ===
    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(
            params=temp_params,
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(
            params=temp_params,
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    # === LR scheduler ===
    cfg.total_batch_size = sum(cfg.batch_size) * world_size
    cfg.warmup_step = sum(cfg.num_image) // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = sum(cfg.num_image) // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)

    # === Resume checkpoint ===
    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])

        for i in range(num_dataset):
            key = f"{i}_state_dict_softmax_fc"
            module_partial_fc_dict[i].load_state_dict(dict_checkpoint[key])

        del dict_checkpoint

    # === Logging ===
    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    # Use first dataset's path for verification bins (all datasets share same val targets)
    val_prefix = cfg.rec[0] if isinstance(cfg.rec, list) else cfg.rec
    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=val_prefix,
        summary_writer=summary_writer, wandb_logger=wandb_logger
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.total_batch_size,
        start_step=global_step,
        writer=summary_writer
    )

    # Per-dataset + combined loss meters
    loss_am_dict = {"all": AverageMeter()}
    for i in range(num_dataset):
        loss_am_dict[i] = AverageMeter()

    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    # === Training loop ===
    for epoch in range(start_epoch, cfg.num_epoch):

        # Set epoch for all samplers
        len_batch_list = []
        for i in range(num_dataset):
            if isinstance(train_loader_dict[i], DataLoader):
                train_loader_dict[i].sampler.set_epoch(epoch)
            len_batch_list.append(len(train_loader_dict[i]))

        max_len = max(len_batch_list)
        logging.info(f"epoch {epoch} max_batch_len: {max_len}")

        for data_batch_index in range(max_len):
            global_step += 1

            # --- Gather batches from all datasets ---
            img_list = []
            local_labels_dict = {}
            for i in range(num_dataset):
                try:
                    img_batch, labels_batch = train_loader_dict[i].next_item(data_batch_index)
                except StopIteration:
                    # Shorter dataset exhausted -> recreate and fetch index 0
                    train_loader_dict[i] = get_dataloader(
                        cfg.rec[i], local_rank, cfg.batch_size[i],
                        cfg.dali, cfg.dali_aug, cfg.seed, cfg.num_workers)
                    if isinstance(train_loader_dict[i], DataLoader):
                        train_loader_dict[i].sampler.set_epoch(epoch)
                    img_batch, labels_batch = train_loader_dict[i].next_item(0)
                img_list.append(img_batch)
                local_labels_dict[i] = labels_batch

            # --- Single forward through shared backbone ---
            img = torch.cat(img_list, 0)
            local_embeddings_all = backbone(img)

            # --- Split embeddings back to per-dataset ---
            split_sizes = [img_list[i].shape[0] for i in range(num_dataset)]
            local_embeddings_list = torch.split(local_embeddings_all, split_sizes, dim=0)

            # --- Per-dataset PartialFC forward ---
            loss_dict = {}
            loss_dict_with_weight = {}
            for i in range(num_dataset):
                loss_i: torch.Tensor = module_partial_fc_dict[i](
                    local_embeddings_list[i],
                    local_labels_dict[i]
                )
                loss_dict[i] = loss_i
                loss_dict_with_weight[i] = loss_i * cfg.loss_w[i]

            loss = sum(loss_dict_with_weight.values())

            # --- Backward + gradient accumulation ---
            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

            # --- Logging ---
            with torch.no_grad():
                # Per-dataset loss logging (each meter is independent, reset after log)
                for i in range(num_dataset):
                    loss_am_dict[i].update(loss_dict[i].item(), 1)
                    callback_logging.self_callback_logging(
                        i, global_step, loss_am_dict[i], epoch,
                        cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                # Combined loss logging
                loss_am_dict["all"].update(loss.item(), 1)
                callback_logging(
                    global_step, loss_am_dict["all"], epoch,
                    cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                # WandB per-step logging
                if wandb_logger:
                    wandb_log = {
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am_dict["all"].avg,
                        'Process/Step': global_step,
                        'Process/Epoch': epoch,
                    }
                    for i in range(num_dataset):
                        wandb_log[f'Loss/dataset_{i}'] = loss_am_dict[i].avg
                    wandb_logger.log(wandb_log)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        # --- Per-epoch checkpoint save ---
        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict(),
            }
            for i in range(num_dataset):
                checkpoint[f"{i}_state_dict_softmax_fc"] = module_partial_fc_dict[i].state_dict()
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0 and epoch % 5 == 0:
            path_module = os.path.join(cfg.output, str(epoch) + "_model.pt")
            torch.save(backbone.module.state_dict(), path_module)

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type='model')
                model.add_file(path_module)
                wandb_logger.log_artifact(model)

        if cfg.dali:
            for i in range(num_dataset):
                train_loader_dict[i].reset()

    # Final model save
    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)

        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_module)
            wandb_logger.log_artifact(model)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Multi-Dataset Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())
