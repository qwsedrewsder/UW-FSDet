_base_ = ['DINO_4scale.py']

num_classes = 4
dn_labelbook_size = 5

# project protocol: fixed 416 input (instead of official multi-scale 480-800)
data_aug_scales = [416]
data_aug_max_size = 416
data_aug_scales2_resize = [416]
data_aug_scales2_crop = [384, 416]

save_checkpoint_interval = 4
