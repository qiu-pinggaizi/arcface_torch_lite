from easydict import EasyDict as edict

config = edict()

# Margin Base Softmax
config.margin_list = (1.0, 0.0, 0.4)
config.network = "mbf_v3_se"
config.resume = False
config.save_all_states = True
config.output = None

config.embedding_size = 512

# Partial FC
config.sample_rate = 0.9
config.interclass_filtering_threshold = 0

config.fp16 = True

# === Multi-Dataset Fields ===
config.rec = [
    "/ipcdata-bj/data/jinj/glint360k",
    "/ipcdata-bj/data/jinj/faces_umd/faces_umd",
]
config.num_classes = [360232, 8277]
config.num_image = [17091657, 811440]
config.batch_size = [128, 128]
config.loss_w = [0.7, 0.3]

# === Distillation Fields ===
config.distill = True
config.teacher_network = "r100"
config.teacher_checkpoint = "/ipcdata-bj/data/jinj/face_rec_train_result/glint360k_r100/model.pt"
config.distill_alpha = 0.5
config.distill_loss_type = "cosine"

# Optimizer
config.optimizer = "sgd"
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 1e-4

config.verbose = 2000
config.frequent = 10

# For Large Scale Dataset
config.dali = False
config.dali_aug = False

# Gradient ACC
config.gradient_acc = 4

# Setup seed
config.seed = 2048

# DataLoader num_workers
config.num_workers = 2

config.num_epoch = 10
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

# WandB Logger
config.wandb_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
config.suffix_run_name = None
config.using_wandb = False
config.wandb_entity = "entity"
config.wandb_project = "project"
config.wandb_log_all = True
config.save_artifacts = False
config.wandb_resume = False
