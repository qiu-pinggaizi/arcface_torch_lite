from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r34"
config.resume = None
config.output = "output/"
config.embedding_size = 512
config.sample_rate = 0.6
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 256
config.lr = 0.1
config.verbose = 5000
config.dali = False
config.save_all_states = True
config.frequent = 60

config.rec = "/path/to/glint360k"
config.num_classes = 360232
config.num_image = 17091657
config.num_epoch = 100
config.warmup_epoch = 1
config.gradient_acc = 2
config.val_targets = ['lfw','vgg2_fp',"agedb_30","calfw","cfp_ff","cplfw","cfp_fp"]
# config.val_targets = ['lfw']
