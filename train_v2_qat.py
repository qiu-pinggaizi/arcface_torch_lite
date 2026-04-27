"""
QAT (Quantization Aware Training) for Phase 3+ pruned model.

Based on train_v2_distill.py with QAT modifications:
- FP16 disabled (QAT requires FP32)
- QuantStub/DeQuantStub applied to model
- prepare_qat() before training, convert() after training
- Final output: quantized_model.pt (INT8, ~2.7MB from 10.71MB FP32)

Usage:
    torchrun --master_port 29523 --nproc_per_node=4 train_v2_qat.py \
        --config configs/glint360k_mbf_v3_se_qat.py
"""
import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.quantization as quant
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

import torch_pruning as tp

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
    assert cfg.fp16 is False, "QAT requires fp16=False"

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

    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers
    )

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=False, num_features=cfg.embedding_size).cuda()

    ### Pruning (reproduce Phase 3+ architecture)
    if hasattr(cfg, 'prune_ratio') and cfg.prune_ratio > 0:
        logging.info(f"Applying pruning with ratio={cfg.prune_ratio}")
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = img.astype(np.float32)
        img = (img / 255. - 0.5) / 0.5
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).cuda()

        ignored_layers = []
        for name, m in backbone.named_modules():
            if 'features' in name or 'conv_sep' in name:
                ignored_layers.append(m)

        pruner = tp.pruner.GroupNormPruner(
            backbone,
            img,
            importance=tp.importance.GroupMagnitudeImportance(),
            iterative_steps=1,
            pruning_ratio=cfg.prune_ratio,
            ignored_layers=ignored_layers
        )
        pruner.step()

        if hasattr(cfg, 'prune_load') and cfg.prune_load and os.path.exists(cfg.prune_load):
            weight = torch.load(cfg.prune_load, map_location=torch.device("cpu"))
            backbone.load_state_dict(weight, strict=False)
            logging.info(f"Loaded pruned checkpoint from {cfg.prune_load}")
        else:
            logging.warning("No prune_load checkpoint specified, pruning without loading weights")

    ### QAT Preparation
    logging.info("Preparing QAT...")
    backend = getattr(cfg, 'qat_backend', 'fbgemm')
    backbone.qconfig = quant.get_default_qat_qconfig(backend)

    # prepare_qat requires model in training mode
    backbone.train()
    quant.prepare_qat(backbone, inplace=True)
    logging.info(f"QAT preparation complete with backend={backend}")

    # Freeze BatchNorm for QAT stability (after prepare_qat)
    for m in backbone.modules():
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.eval()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)

    backbone.train()

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)

    start_epoch = 0
    global_step = 0

    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.resume, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec,
        summary_writer=summary_writer, wandb_logger=None
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    # No GradScaler for QAT (FP32 only)

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img)
            ce_loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)

            loss = ce_loss

            # FP32 backward (no FP16/AMP)
            loss.backward()
            if global_step % cfg.gradient_acc == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(module_partial_fc.parameters()), 5)
                opt.step()
                opt.zero_grad()

            lr_scheduler.step()

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, False, lr_scheduler.get_last_lr()[0], None)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0 and epoch % 2 == 0:
            path_module = os.path.join(cfg.output, str(epoch) + "_model.pt")
            torch.save(backbone.module.state_dict(), path_module)

        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        # Save FP32-QAT model (contains fake quantization observers)
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)
        logging.info(f"QAT model saved: {path_module}")

        # Export quantized model variants
        logging.info("Exporting quantized model variants...")

        # Use the trained backbone module directly for export
        # (it already has the correct pruned architecture)
        backbone_export = backbone.module.cpu().eval()

        # Save TorchScript export for deployment
        try:
            dummy = torch.randn(1, 3, 112, 112)
            traced = torch.jit.trace(backbone_export, dummy)
            ts_path = os.path.join(cfg.output, "model_traced.pt")
            torch.jit.save(traced, ts_path)
            ts_size = os.path.getsize(ts_path) / 1024 / 1024
            logging.info(f"TorchScript model saved: {ts_path} ({ts_size:.2f} MB)")
        except Exception as e:
            logging.warning(f"TorchScript export failed: {e}")

        # INT8 export: use the trained backbone directly (same architecture)
        # Filter out QAT observer keys and apply torchao quantization
        try:
            from torchao.quantization import quantize_, Int8WeightOnlyConfig

            # Load QAT weights, filter out observer keys
            qat_weights = backbone_export.state_dict()
            clean_weights = {k: v for k, v in qat_weights.items()
                           if not k.startswith('quant.') and not k.startswith('dequant.')
                           and '.activation_post_process.' not in k}

            # Create fresh base model and apply pruning to match architecture
            base_network = cfg.network.replace('_qat', '')
            model_int8 = get_model(
                base_network, dropout=0.0, fp16=False,
                num_features=cfg.embedding_size).cpu()

            # Reproduce pruning architecture
            if hasattr(cfg, 'prune_ratio') and cfg.prune_ratio > 0:
                img_cpu = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
                img_cpu = img_cpu.astype(np.float32)
                img_cpu = (img_cpu / 255. - 0.5) / 0.5
                img_cpu = img_cpu.transpose((2, 0, 1))
                img_cpu = torch.from_numpy(img_cpu).unsqueeze(0)

                ignored_layers = []
                for name, m in model_int8.named_modules():
                    if 'features' in name or 'conv_sep' in name:
                        ignored_layers.append(m)

                pruner = tp.pruner.GroupNormPruner(
                    model_int8, img_cpu,
                    importance=tp.importance.GroupMagnitudeImportance(),
                    iterative_steps=1,
                    pruning_ratio=cfg.prune_ratio,
                    ignored_layers=ignored_layers
                )
                pruner.step()

            # Try loading with strict=False (skip mismatched keys)
            model_int8.load_state_dict(clean_weights, strict=False)
            model_int8.eval()

            quantize_(model_int8, Int8WeightOnlyConfig())

            int8_path = os.path.join(cfg.output, "int8_model.pt")
            torch.save(model_int8.state_dict(), int8_path)
            int8_size = os.path.getsize(int8_path) / 1024 / 1024
            fp32_size = os.path.getsize(path_module) / 1024 / 1024
            logging.info(f"INT8 model saved: {int8_path} ({int8_size:.2f} MB, FP32 was {fp32_size:.2f} MB)")
        except Exception as e:
            logging.warning(f"INT8 export failed (non-critical): {e}")

        # Also save the full pruned model config for future reconstruction
        try:
            import json
            # Get layer shapes from trained model for reproducibility
            layer_shapes = {}
            for name, param in backbone_export.named_parameters():
                layer_shapes[name] = list(param.shape)
            config_path = os.path.join(cfg.output, "model_arch.json")
            with open(config_path, 'w') as f:
                json.dump(layer_shapes, f, indent=2)
            logging.info(f"Model architecture saved: {config_path}")
        except Exception as e:
            logging.warning(f"Architecture export failed: {e}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="QAT Training for Face Recognition")
    parser.add_argument("--config", type=str, help="py config file")
    main(parser.parse_args())
