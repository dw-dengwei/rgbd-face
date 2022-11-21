class BaseConfig:
    """env"""
    gpu = [1, 2]
    precision = 16
    num_workers = 10
    pin_memory = True
    persistent_workers = True
    random_seed = 42

    """train"""
    epoch = 30
    batch_size = 512
    learning_rate = 3e-4
    valid_check_interval = 0.5

    """model"""
    num_classes = 600
    backbone = "resnet18"
    rgb_weight = 0.5
    arcface_margin = 0.3
    out_features = 1024
    reduction = 16
    lr_reduce_epoch = [6, 10, 17]

    """data"""
    train_data_root = "/home/dw/vgg"
    train_rgb_dir = "crop"
    train_dzyx_dir = "depth_normal"
    train_ids = 468
    train_using_modal = ("rgb", "normal")

    valid_data_root = "/home/dw/rgbd/Texas"
    valid_rgb_dir = "aligned"
    valid_dzyx_dir = "depth_normal"
    valid_ids = 118
    valid_using_modal = train_using_modal
