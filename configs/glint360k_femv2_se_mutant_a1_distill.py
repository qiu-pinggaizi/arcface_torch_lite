from easydict import EasyDict as edict

config = edict()

# 模型配置
config.network = "femv2_se_mutant_a1"  # student
config.output = "output/"
config.embedding_size = 512

# 蒸馏配置
config.distill = True
config.teacher_network = "r100"
config.teacher_checkpoint = "/path/to/teacher/model.pt"
config.teacher_prune_ratio = None
config.distill_alpha = 0.5
config.distill_loss_type = "cosine"

# 训练配置
config.optimizer = "adamw"
config.lr = 0.001
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 64
config.gradient_acc = 1
config.num_epoch = 50
config.warmup_epoch = 5
config.fp16 = True

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
config.verbose = 5000
config.frequent = 60
config.dali = False
config.dali_aug = False
config.resume = False
config.save_all_states = True
config.interclass_filtering_threshold = 0
