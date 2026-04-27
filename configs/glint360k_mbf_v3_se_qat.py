from easydict import EasyDict as edict

config = edict()

# 模型配置
config.network = "mbf_v3_se_qat"  # QAT 版本
config.output = "output/"
config.embedding_size = 512

# 剪枝配置（必须与 Phase 3+ 一致）
config.prune_ratio = 0.2
config.prune_load = "/path/to/pruned/model.pt"

# QAT 配置
config.qat_backend = "fbgemm"  # CPU 推理用 fbgemm，移动端用 qnnpack

# 训练配置（QAT 微调）
config.optimizer = "sgd"
config.lr = 0.01                # 降低学习率（原 0.1）
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 256
config.gradient_acc = 2
config.num_epoch = 10           # 短期微调（原 90）
config.warmup_epoch = 0         # QAT 不需要 warmup
config.fp16 = False             # QAT 要求 FP32

# 数据配置
config.rec = "/path/to/glint360k"
config.num_classes = 360232
config.num_image = 17091657
config.num_workers = 2
config.sample_rate = 0.9

# 损失配置
config.margin_list = (1.0, 0.0, 0.4)

# 验证配置
config.val_targets = ['lfw', 'vgg2_fp', 'agedb_30', 'calfw', 'cfp_ff', 'cplfw', 'cfp_fp']

# 其他
config.seed = 2048
config.verbose = 2000           # 更频繁验证（原 5000）
config.frequent = 60
config.dali = False
config.dali_aug = False
config.resume = False
config.save_all_states = True
config.interclass_filtering_threshold = 0
