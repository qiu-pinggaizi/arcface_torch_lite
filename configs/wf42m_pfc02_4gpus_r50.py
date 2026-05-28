from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r50"
config.resume = False
config.output = "/ipcdata-bj/data/jinj/face_rec_train_result/wf42m_r50_4gpu"
config.embedding_size = 512
config.sample_rate = 0.2
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 256
config.lr = 0.2
config.verbose = 10000
config.dali = False
config.save_all_states = True

config.rec = "/ipcdata-bj/data/jinj/WebFace260M/新加内容"
config.num_classes = 2059906
config.num_image = 42474558
config.num_epoch = 30
config.warmup_epoch = 2
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
