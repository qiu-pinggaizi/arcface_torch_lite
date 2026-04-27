import sys, os, json, time, torch, math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = r'/ipcdata-tj/duguohui/face_recognition/ModelEvolutionEngine/datasets/06_2d_feature_extraction'
if not os.path.isdir(SHARED_DIR):
    # fallback: 向上遍历查找包含 'datasets' 子目录的 automl 根目录
    _cur = os.path.abspath(SCRIPT_DIR)
    _automl_root = None
    while _cur != os.path.dirname(_cur):
        if os.path.isdir(os.path.join(_cur, 'datasets')):
            _automl_root = _cur
            break
        _cur = os.path.dirname(_cur)
    if _automl_root:
        # 推断 module name：从 model_ir.json 或向上查找包含 config.yaml 的目录
        _module_name = ''
        _ir_path = os.path.join(SCRIPT_DIR, 'model_ir.json')
        if os.path.exists(_ir_path):
            try:
                with open(_ir_path) as _f:
                    _ir = json.load(_f)
                _module_name = _ir.get('metadata', {}).get('module_name', '')
            except Exception:
                pass
        if not _module_name:
            # 向上查找包含 config.yaml 的目录名
            _d = os.path.abspath(SCRIPT_DIR)
            while _d != os.path.dirname(_d):
                if os.path.exists(os.path.join(_d, 'config.yaml')):
                    _module_name = os.path.basename(_d)
                    break
                _d = os.path.dirname(_d)
        if _module_name:
            SHARED_DIR = os.path.join(_automl_root, 'datasets', _module_name)
        else:
            SHARED_DIR = os.path.join(_automl_root, 'datasets')
        SHARED_DIR = os.path.abspath(SHARED_DIR)
    if not os.path.isdir(SHARED_DIR):
        print(f"WARNING: Cannot find datasets dir, SHARED_DIR={SHARED_DIR}")
sys.path.insert(0, SHARED_DIR)

# 让 train_utils 能找到 train_config.json / model_ir.json
os.environ['TRAIN_SCRIPT_DIR'] = SCRIPT_DIR

# 导入训练工具和数据集
from dataset import get_train_test_datasets, ArcFaceHead, TrainingWrapper
from train_utils import _instantiate_model, _kaiming_init, _extract_features

# 导入模型
import importlib
model_py = os.path.join(SCRIPT_DIR, 'model.py')
spec = importlib.util.spec_from_file_location("candidate_model", model_py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# 从model_ir.json获取目标class名，优先匹配
_TARGET_CLASS = None
_ir_path = os.path.join(SCRIPT_DIR, 'model_ir.json')
if os.path.exists(_ir_path):
    with open(_ir_path) as _f:
        _ir = json.load(_f)
    _TARGET_CLASS = _ir.get('metadata', {}).get('original_class')

MODEL_CLASS = None
for name in dir(mod):
    obj = getattr(mod, name)
    if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
        if _TARGET_CLASS and obj.__name__ == _TARGET_CLASS:
            MODEL_CLASS = obj
            break
        if obj.__name__ == 'GeneratedModel':
            MODEL_CLASS = obj
        elif MODEL_CLASS is None:
            MODEL_CLASS = obj

if MODEL_CLASS is None:
    print("ERROR: No nn.Module subclass found in model.py")
    sys.exit(1)

FEAT_DIM = 256
NUM_CLASSES = 300
BATCH_SIZE = 64
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
PATIENCE = 50
GRAD_CLIP = 5.0
ARCFACE_SCALE = 30.0
ARCFACE_MARGIN = 0.5


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
    train_dataset, test_dataset, actual_num_classes = get_train_test_datasets()
    NUM_CLASSES_ACTUAL = actual_num_classes if actual_num_classes > 0 else NUM_CLASSES
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True, drop_last=True)
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}, Classes: {NUM_CLASSES_ACTUAL}")

    # 模型 — 智能实例化 + Kaiming初始化（从头训练）
    backbone = _instantiate_model(MODEL_CLASS)
    _kaiming_init(backbone)
    print("Model initialized with Kaiming initialization")
    # 检查backbone输出维度
    backbone.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 1, 112, 112)
        out = _extract_features(backbone, dummy)
        actual_feat_dim = out.shape[-1]
        print(f"Backbone output dim: {actual_feat_dim}")
    backbone.train()

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
    start_time = time.time()

    # 早停
    early_stop_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            backbone, arcface_head, train_loader,
            optimizer, optimizer_arcface, criterion, device
        )
        scheduler.step()
        scheduler_arcface.step()

        # 每5个epoch评估Rank-1（或最后一个epoch）
        test_rank1 = 0.0
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            test_rank1 = evaluate_rank1(backbone, test_loader, device)
            print(f"Epoch {epoch}/{NUM_EPOCHS} - train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"test_rank1={test_rank1:.2f}% lr={scheduler.get_last_lr()[0]:.6f}")
        else:
            print(f"Epoch {epoch}/{NUM_EPOCHS} - train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_rank1': test_rank1,
            'lr': scheduler.get_last_lr()[0]
        })

        if test_rank1 > best_rank1:
            best_rank1 = test_rank1
            best_epoch = epoch
            early_stop_counter = 0
            os.makedirs(os.path.join(SCRIPT_DIR, 'weights'), exist_ok=True)
            torch.save(backbone.state_dict(), os.path.join(SCRIPT_DIR, 'weights', 'best.pth'))
        elif test_rank1 > 0:
            early_stop_counter += 5  # 每次评估间隔5个epoch
            if early_stop_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {early_stop_counter} epochs, patience={PATIENCE})")
                break

    # 保存指标
    training_time = time.time() - start_time
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
        'training_time_seconds': round(training_time, 1),
    }
    with open(os.path.join(SCRIPT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(SCRIPT_DIR, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nBest Rank-1: {best_rank1:.2f}% at epoch {best_epoch}")
    print(f"Metrics saved.")


if __name__ == '__main__':
    main()
