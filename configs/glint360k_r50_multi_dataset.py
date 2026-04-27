from easydict import EasyDict as edict

config = edict()

# Margin Base Softmax
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r50"
config.resume = False
config.save_all_states = True
config.output = None

config.embedding_size = 512

# Partial FC
config.sample_rate = 0.1
config.interclass_filtering_threshold = 0

config.fp16 = True

# === Multi-Dataset Fields (list-type, one per dataset) ===
config.rec = [
    "/path/to/glint360k",
    "/path/to/second_dataset",
]
config.num_classes = [360232, 585688]
config.num_image = [17091657, 3227342]
config.batch_size = [128, 128]
config.loss_w = [0.25, 0.75]

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
config.gradient_acc = 1

# Setup seed
config.seed = 2048

# DataLoader num_workers
config.num_workers = 2

config.num_epoch = 20
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
