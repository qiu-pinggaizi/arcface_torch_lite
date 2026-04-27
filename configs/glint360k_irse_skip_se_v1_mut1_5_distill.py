from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "irse_skip_se_v1_mut1_5"  # student
config.resume = False
config.output = "/ipcdata-bj/data/jinj/face_rec_train_result/irse_skip_se_v1_mut1_5_distill"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 256
config.lr = 0.1
config.verbose = 2000
config.dali = False

# 蒸馏配置
config.distill = True
config.teacher_network = "r100"
config.teacher_checkpoint = "/ipcdata-bj/data/jinj/face_rec_train_result/glint360k_r100/model.pt"
config.teacher_prune_ratio = None
config.distill_alpha = 0.5
config.distill_loss_type = "cosine"

config.rec = "/ipcdata-tj/data/jinj/glint360k"
config.num_classes = 360232
config.num_image = 17091657
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
