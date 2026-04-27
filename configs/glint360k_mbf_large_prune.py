from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "mbf_large"
config.output = "/ipcdata-bj/data/jinj/face_rec_train_result/mbf_large_0.4ir_prune"

config.embedding_size = 512
config.sample_rate = 0.9
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 256
config.lr = 0.1
config.verbose = 5000
config.dali = False
config.frequent = 60
config.save_all_states = True
config.gradient_acc = 2
config.resume = "/ipcdata-bj/data/jinj/face_rec_train_result/mbf_large_0.3ir_prune"
# config.wandb_key = "51157ef3d01a6b1cf6ddcc79ce57b03fe67aded1"
# config.using_wandb = 1

config.rec = "/ipcdata-tj/data/jinj/glint360k"
config.num_classes = 360232
config.num_image = 17091657
config.num_epoch = 75
config.warmup_epoch = 1
config.val_targets = ['lfw','vgg2_fp',"agedb_30","calfw","cfp_ff","cplfw","cfp_fp"]
# config.val_targets = ['lfw']
# config.only_eval = True
