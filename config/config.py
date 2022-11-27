class BaseConfig:
    """env"""
    gpu = [1, 3]
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
    momentum = 0.9
    weight_decay = 0.0005
    warmup_ratio = 0.05

    """model"""
    num_classes = 600
    backbone = "resnet18"
    rgb_weight = 0.5
    arcface_margin = 0.3
    out_features = 1024
    reduction = 16
    lr_reduce_epoch = [6, 10, 17]
    alpha = 1
    beta = 2
    gamma = 3
    lambda_1 = 0.5
    lambda_2 = 0.05
    rgb_in_channels = 3
    depth_in_channels = 3

    """training data"""
    train_data_root = "/home/dw/vgg"
    train_rgb_dir = "aligned"
    train_dzyx_dir = "3ddfav2_depth_normal"
    train_ids = 468
    train_using_modal = ("rgb", "normal")
    using_test = 'lock3dface'
    single_modal = 'depth'

    """Texas data"""
    texas_data_root = "/home/dw/rgbd/Texas"
    texas_rgb_dir = "aligned"
    texas_dzyx_dir = "depth_normal"
    texas_ids = 118
    texas_using_modal = train_using_modal

    """Lock3DFace data"""
    lock3dface_datainfo = "/home/dw/code/rgbd-face/util/lock3dface_info.csv"
    lock3dface_data_root = "/home/dw/data/lock3dface/test/"
    lock3dface_rgb_dir = "aligned"
    lock3dface_dzyx_dir = "depth"
    lock3dface_using_modal = train_using_modal
