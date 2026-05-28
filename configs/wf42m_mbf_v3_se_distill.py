from easydict import EasyDict as edict

config = edict()

# 模型配置
config.network = "mbf_v3_se"  # student
config.output = "/ipcdata-bj/data/jinj/face_rec_train_result/wf42m_mbf_v3_se_distill"
config.embedding_size = 512

# 蒸馏配置
config.distill = True
config.teacher_network = "r50"
config.teacher_checkpoint = "/ipcdata-bj/data/jinj/face_rec_train_result/wf42m_r50_4gpu/model.pt"
config.teacher_prune_ratio = None
config.distill_alpha = 0.5
config.distill_loss_type = "cosine"

# 训练配置
config.optimizer = "sgd"
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 256
config.gradient_acc = 2
config.num_epoch = 30
config.warmup_epoch = 1
config.fp16 = True

# 数据配置
config.rec = "/ipcdata-bj/data/jinj/WebFace260M/新加内容"
config.num_classes = 2059906
config.num_image = 42474558
config.sample_rate = 0.2

# 损失配置
config.margin_list = (1.0, 0.0, 0.4)

# 验证配置
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

# 其他
config.seed = 2048
config.verbose = 5000
config.frequent = 60
config.dali = False
config.resume = False
config.save_all_states = True
