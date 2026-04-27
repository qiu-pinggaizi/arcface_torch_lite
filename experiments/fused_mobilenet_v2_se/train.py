import sys, os, json, torch, math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..', '..', '..'))
MODULE_DIR = os.path.join(PROJECT_ROOT, '06_2d_feature_extraction')
TRAINING_DIR = os.path.join(PROJECT_ROOT, 'training_generated', '06_2d_feature_extraction')
sys.path.insert(0, os.path.abspath(MODULE_DIR))
sys.path.insert(0, os.path.abspath(TRAINING_DIR))

# 导入模型
import importlib
model_py = os.path.join(SCRIPT_DIR, 'model.py')
spec = importlib.util.spec_from_file_location("candidate_model", model_py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# 找到nn.Module子类（优先GeneratedModel，否则最后一个）
MODEL_CLASS = None
for name in dir(mod):
    obj = getattr(mod, name)
    if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
        if obj.__name__ == 'GeneratedModel':
            MODEL_CLASS = obj
            break
        MODEL_CLASS = obj  # fallback: 最后一个

if MODEL_CLASS is None:
    print("ERROR: No nn.Module subclass found in model.py")
    sys.exit(1)

from dataset import get_train_val_datasets, ArcFaceHead, TrainingWrapper

FEAT_DIM = 256
NUM_CLASSES = 300
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 50
GRAD_CLIP = 1.0
ARCFACE_SCALE = 30.0
ARCFACE_MARGIN = 0.5

def _instantiate_model(model_cls):
    """智能实例化模型：先尝试无参，失败则检查签名"""
    import inspect
    sig = inspect.signature(model_cls.__init__)
    params = [p for p in sig.parameters.values() if p.name != 'self' and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]

    # 如果无额外参数，直接无参实例化
    if not params:
        return model_cls()

    # 尝试用默认值实例化（所有参数都有默认值的话）
    if all(p.default is not inspect.Parameter.empty for p in params):
        return model_cls()

    # 需要推断参数：从config.json读取
    config_path = os.path.join(SCRIPT_DIR, 'train_config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        # 尝试用config中的参数实例化
        kwargs = {{}}
        for p in params:
            if p.name in cfg:
                kwargs[p.name] = cfg[p.name]
        if len(kwargs) == len(params):
            return model_cls(**kwargs)

    # 回退：用IR中的信息推断
    ir_path = os.path.join(SCRIPT_DIR, 'model_ir.json')
    if os.path.exists(ir_path):
        with open(ir_path) as f:
            ir = json.load(f)
        arch = ir.get('architecture', {{}})
        backbone_cfg = arch.get('backbone', {{}})
        channels = backbone_cfg.get('channels', [64, 64, 128, 256, 512])
        feat_dim = arch.get('head', {{}}).get('feat_dim', 256)

        # 常见参数映射
        defaults = {{
            'in_ch': channels[0], 'out_ch': channels[-1],
            'num_classes': 100, 'embedding_size': feat_dim, 'embed_dim': feat_dim,
            'feat_dim': feat_dim, 'num_features': channels[-1],
            'input_channels': 1, 'in_channels': 1, 'out_channels': channels[-1],
            'stride': 1, 'expand_ratio': 4, 'use_se': True,
            'config': None,
        }}
        kwargs = {{}}
        for p in params:
            if p.default is not inspect.Parameter.empty:
                kwargs[p.name] = p.default
            elif p.name in defaults:
                kwargs[p.name] = defaults[p.name]
            else:
                print(f"WARNING: cannot infer param '{{p.name}}' for {{model_cls.__name__}}, using 1")
                kwargs[p.name] = 1
        return model_cls(**kwargs)

    raise TypeError(f"Cannot instantiate {{model_cls.__name__}}: requires params {{[p.name for p in params]}}")


def _forward_model(model, x):
    """自适应forward：从backbone或TrainingWrapper中提取embedding。

    TrainingWrapper返回 (logits, features)，取第2个元素。
    普通backbone直接返回embedding tensor（或单元素tuple）。
    规则：
    - 如果返回tuple且长度>=2: 取倒数第1个（features/embedding）
    - 如果返回单个tensor: 直接返回
    - 特殊：MobileFaceNetMagFace256D返回(embedding, quality)，取第0个
    """
    out = model(x)
    if isinstance(out, tuple):
        # tuple返回：通常 (embedding, quality) 或 (logits, features)
        # 取第一个元素（embedding/logits）还是第二个（quality/features）？
        # 标准TrainingWrapper: (logits, features) → 取features
        # 标准backbone: (embedding, ...) → 取embedding
        # 由于无法区分，保守取第0个，因为：
        # - TrainingWrapper的train_epoch不需要logits（loss基于features）
        # - evaluate_epoch会单独处理logits
        return out[0]
    return out


def _extract_features(backbone, x):
    """从backbone提取embedding（处理tuple返回）。

    backbone可能返回单tensor或tuple。
    如果是tuple，最后一个元素通常是features/embedding。
    """
    out = backbone(x)
    if isinstance(out, tuple):
        return out[-1]
    return out



# ========== ArcFaceHead（内联，与v4种子模型完全一致） ==========
# 注意：dataset.py 中的 ArcFaceHead 已与v4一致，这里从 dataset 导入


def train_epoch(backbone, arcface_head, dataloader, optimizer, optimizer_arcface, criterion, device):
    """训练一个epoch — 对齐v4种子模型的双优化器方式。"""
    backbone.train()
    arcface_head.train()

    total_loss = 0.0
    correct = 0
    total = 0
    use_bn_eval = False

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device).long()

        optimizer.zero_grad()
        optimizer_arcface.zero_grad()

        # 检测 BN 1x1 问题（仅第一个batch探测）
        if not use_bn_eval and total == 0:
            try:
                features = _extract_features(backbone, images)
            except ValueError as e:
                if 'more than 1 value per channel' in str(e):
                    use_bn_eval = True
                    backbone.eval()
                    arcface_head.eval()
                    torch.set_grad_enabled(True)
                    features = _extract_features(backbone, images)
                else:
                    raise
        else:
            features = _extract_features(backbone, images)

        logits = arcface_head(features, labels)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=GRAD_CLIP)
        torch.nn.utils.clip_grad_norm_(arcface_head.parameters(), max_norm=GRAD_CLIP)

        optimizer.step()
        optimizer_arcface.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_rank1(backbone, dataloader, device):
    """Rank-1 评估 — 与v4种子模型完全一致（gallery/probe 余弦相似度匹配）。"""
    backbone.eval()

    gallery_features = []
    gallery_labels = []
    probe_features = []
    probe_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        features = _extract_features(backbone, images)
        features = F.normalize(features, p=2, dim=1)

        for i in range(features.size(0)):
            if i % 2 == 0:
                gallery_features.append(features[i].cpu())
                gallery_labels.append(labels[i].item())
            else:
                probe_features.append(features[i].cpu())
                probe_labels.append(labels[i].item())

    if len(gallery_features) == 0 or len(probe_features) == 0:
        return 0.0

    gallery_features = torch.stack(gallery_features)
    probe_features = torch.stack(probe_features)

    similarity = torch.mm(probe_features, gallery_features.t())
    _, indices = similarity.max(dim=1)
    correct = 0
    for i, idx in enumerate(indices):
        if gallery_labels[idx] == probe_labels[i]:
            correct += 1

    rank1 = 100.0 * correct / len(probe_labels)
    return rank1


def evaluate_loss(backbone, arcface_head, dataloader, criterion, device):
    """评估loss（用于训练过程中的监控）。"""
    backbone.eval()
    arcface_head.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).long()
            features = _extract_features(backbone, images)
            logits = arcface_head(features, labels)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            total += labels.size(0)
    avg_loss = total_loss / max(total, 1)
    return avg_loss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 数据集 — 使用LAMP-HQ真实数据
    train_dataset, val_dataset, actual_num_classes = get_train_val_datasets()
    NUM_CLASSES_ACTUAL = actual_num_classes if actual_num_classes > 0 else NUM_CLASSES
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Classes: {NUM_CLASSES_ACTUAL}")

    # 模型 — 智能实例化
    backbone = _instantiate_model(MODEL_CLASS)
    # 检查backbone输出维度
    with torch.no_grad():
        dummy = torch.randn(1, 1, 112, 112)
        out = _extract_features(backbone, dummy)
        actual_feat_dim = out.shape[-1]
        print(f"Backbone output dim: {actual_feat_dim}")

    backbone = backbone.to(device)

    # ArcFace分类头（与v4完全一致）
    arcface_head = ArcFaceHead(
        feat_dim=actual_feat_dim,
        num_classes=NUM_CLASSES_ACTUAL,
        scale=ARCFACE_SCALE,
        margin=ARCFACE_MARGIN,
    ).to(device)

    # 双优化器（与v4一致）
    optimizer = optim.AdamW(backbone.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer_arcface = optim.AdamW(arcface_head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scheduler_arcface = CosineAnnealingLR(optimizer_arcface, T_max=NUM_EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    params_count = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Params: {params_count:,}")

    # 计算FLOPs（通过hook统计Conv2d和Linear）
    flops_count = 0
    try:
        flops_list = []
        def _flops_hook(module, inp, out):
            try:
                if isinstance(module, nn.Conv2d):
                    if out.dim() == 4:
                        out_h, out_w = out.shape[2], out.shape[3]
                    elif out.dim() == 2:
                        out_h, out_w = 1, 1
                    else:
                        return
                    f = 2 * module.in_channels * module.out_channels * out_h * out_w * module.kernel_size[0] * module.kernel_size[1] // module.groups
                    flops_list.append(f)
                elif isinstance(module, nn.Linear):
                    f = 2 * module.in_features * module.out_features
                    flops_list.append(f)
            except Exception:
                pass
        hooks = []
        for m in backbone.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                hooks.append(m.register_forward_hook(_flops_hook))
        with torch.no_grad():
            backbone(torch.randn(1, 1, 112, 112, device=device))
        for h in hooks:
            h.remove()
        flops_count = sum(flops_list)
        print(f"FLOPs: {flops_count:,}")
    except Exception as _flops_err:
        print(f"FLOPs calculation failed: {_flops_err}")

    # 兜底：如果FLOPs为0，用thop库或逐层估算
    if flops_count == 0:
        try:
            from thop import profile
            input_tensor = torch.randn(1, 1, 112, 112, device=device)
            macs, _ = profile(backbone, inputs=(input_tensor,), verbose=False)
            flops_count = int(macs * 2)
            print(f"FLOPs (thop): {flops_count:,}")
        except Exception:
            pass
    if flops_count == 0:
        try:
            # 最后兜底：统计所有Conv2d和Linear的MACs
            total_macs = 0
            x = torch.randn(1, 1, 112, 112, device=device)
            for m in backbone.modules():
                if isinstance(m, nn.Conv2d):
                    if x.dim() == 4:
                        oh, ow = x.shape[2], x.shape[3]
                    else:
                        oh, ow = 1, 1
                    total_macs += m.in_channels * m.out_channels * oh * ow * m.kernel_size[0] * m.kernel_size[1] // m.groups
                elif isinstance(m, nn.Linear):
                    total_macs += m.in_features * m.out_features
            flops_count = total_macs * 2
            print(f"FLOPs (fallback): {flops_count:,}")
        except Exception:
            flops_count = 1  # 避免为0
            print("FLOPs: set to 1 as fallback")

    best_rank1 = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            backbone, arcface_head, train_loader,
            optimizer, optimizer_arcface, criterion, device
        )
        scheduler.step()
        scheduler_arcface.step()

        # 每5个epoch评估Rank-1（或最后一个epoch）
        val_rank1 = 0.0
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            val_rank1 = evaluate_rank1(backbone, val_loader, device)
            print(f"Epoch {epoch}/{NUM_EPOCHS} - train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_rank1={val_rank1:.2f}% lr={scheduler.get_last_lr()[0]:.6f}")
        else:
            print(f"Epoch {epoch}/{NUM_EPOCHS} - train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_rank1': val_rank1,
            'lr': scheduler.get_last_lr()[0]
        })

        if val_rank1 > best_rank1:
            best_rank1 = val_rank1
            best_epoch = epoch
            os.makedirs(os.path.join(SCRIPT_DIR, 'weights'), exist_ok=True)
            torch.save(backbone.state_dict(), os.path.join(SCRIPT_DIR, 'weights', 'best.pth'))

    # 保存指标
    error_rate = 1.0 - best_rank1 / 100.0
    metrics = {
        'accuracy': best_rank1 / 100.0,  # 归一化到 [0,1]
        'rank1': best_rank1,
        'error_rate': error_rate,
        'params': params_count,
        'flops': flops_count,
        'error_rate_x_params': error_rate * params_count,
        'error_rate_x_flops': error_rate * flops_count,
        'best_epoch': best_epoch,
        'epochs_trained': len(history),
    }
    with open(os.path.join(SCRIPT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(SCRIPT_DIR, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nBest Rank-1: {best_rank1:.2f}% at epoch {best_epoch}")
    print(f"Metrics saved.")


if __name__ == '__main__':
    main()
