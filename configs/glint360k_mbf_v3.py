from easydict import EasyDict as edict

config = edict()
config.network = "mbf_v3"
config.output = "output/"
config.embedding_size = 512
config.optimizer = "sgd"
config.margin_list = (1.0, 0.0, 0.4)
config.sample_rate = 0.9
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 256
config.lr = 0.1
config.gradient_acc = 2
config.verbose = 5000
config.frequent = 60
config.dali = False
config.save_all_states = True
config.rec = "/path/to/glint360k"
config.num_classes = 360232
config.num_image = 17091657
config.num_epoch = 90
config.warmup_epoch = 1
config.val_targets = ['lfw', 'vgg2_fp', "agedb_30", "calfw", "cfp_ff", "cplfw", "cfp_fp"]
