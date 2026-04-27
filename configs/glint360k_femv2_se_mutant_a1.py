'''
Author: jinjie xinjie@xiaomi.com
Date: 2026-04-22 17:26:26
LastEditors: jinjie xinjie@xiaomi.com
LastEditTime: 2026-04-22 17:42:00
FilePath: /face_rec/arcface_torch_5max/configs/glint360k_femv2_se_mutant_a1.py
'''
from easydict import EasyDict as edict

config = edict()

# 模型配置
config.network = "femv2_se_mutant_a1"
config.output = "/ipcdata-bj/data/jinj/face_rec_train_result/femv2_se_mutant_a1"
config.embedding_size = 512

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
