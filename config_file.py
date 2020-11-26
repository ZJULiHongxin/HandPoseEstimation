class Config(object):
    root_dir = ''  # r'D:\人手数据集\RHD_v1-1\RHD_published_v2'
    ckpt_dir = 'checkpoints'  # r'D:\人手数据集\RHD_v1-1\RHD_published_v2\checkpoints'
    history_dir = 'history'
    model_name = 'Pose2D Regressor'
    num_train_samples = 0  # will be changed during reading data sets
    num_val_samples = 0

    # settings used for tensorboard
    record_period = 100

    # Hyper-parameters for the training data loader
    heat_map_size = 64
    kernel_size = 7
    sigma = 1
    input_size = 256
    num_epochs = 5
    batch_size = 8
    is_shuffle = False

    # Parameters for the HourGlass-based heat map generator
    stack = 2
    depth = 4
    in_channels = 3
    last_channels = 21

    # Weight coefficients for loss items
    hm_loss_weight = 1
    kps_loss_weight = 1
    # for the model optimizer
    # Adam
    lr = 1e-3
    weight_decay = 1e-5
    step_size = 3
    gamma = 0.1

    # for loss functions
    temperature = 10.0

config = Config()

