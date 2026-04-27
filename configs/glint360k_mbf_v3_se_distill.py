from easydict import EasyDict as edict

config = edict()

# 模型配置
config.network = "mbf_v3_se"  # student
config.output = "/ipcdata-bj/data/jinj/face_rec_train_result/mbf_v3_se_distill"
config.embedding_size = 512

# 蒸馏配置
config.distill = True
config.teacher_network = "mbf_large"
config.teacher_checkpoint = "/ipcdata-bj/data/jinj/face_rec_train_result/mbf_large/model.pt"
config.teacher_prune_ratio = None  # teacher 使用原始 mbf_large，无需剪枝
config.distill_alpha = 0.5  # CE loss 和 distill loss 各占一半
config.distill_loss_type = "cosine"

# 训练配置
config.optimizer = "sgd"
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 256
config.gradient_acc = 2
config.num_epoch = 90
config.warmup_epoch = 1
config.fp16 = True

# 数据配置
config.rec = "/ipcdata-tj/data/jinj/glint360k"
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
config.suffix_run_name = None
config.using_wandb = False
config.wandb_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
config.wandb_entity = "entity"
config.wandb_project = "project"
config.wandb_log_all = True
config.save_artifacts = False
config.wandb_resume = False
