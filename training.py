import os
import shutil
import time
import logging

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import kornia
from kornia.geometry import spatial_soft_argmax2d
from kornia.geometry.subpix import dsnt

from HGFilters import HGFilter
from config_file import config
import metrics
from dataset_utils import hm_dataset

recorder = metrics.MetricsRecorder()
device = metrics.device



def evaluate(val_data_loader, model):
    model.eval()  # model.eval()会仍会执行gradient的计算和存储，只是不进行反传
    recorder.eval()
    with torch.no_grad():  # 停止autograd模块的工作，以起到加速和节省显存的作用，但是并不会影响dropout和batchnorm层的行为
        for i, batch in enumerate(val_data_loader):
            input_img = batch['image'].to(device)  # detach: no-gradient
            hm_groundtruth = batch['heat_map_gn'].to(device)
            output = model(input_img)
            kps_gn = batch['2d_kp_anno'].to(device)
            recorder.process_validation_output(hm_pred=output[0],
                                               hm_gn=hm_groundtruth,
                                               kps_gn=kps_gn)
            del input_img
            del hm_groundtruth
            del output
    return recorder.finish_eval()


def save_checkpoint(state: dict, is_best_HM_loss, is_best_pose2d_loss, is_best_total_loss):
    hm_loss_str = "{:.4f}".format(recorder.val_hm_loss[-1])
    kps_loss_str = "{:.4f}".format(recorder.val_kps_loss[-1])
    total_loss_str = "{:.4f}".format(recorder.val_total_loss[-1])

    filename = os.path.join(config.ckpt_dir,
                            config.model_name + "-ckpt_HMLoss_" + hm_loss_str + "_PoseLoss_" + kps_loss_str + "_TotalLOss_" + total_loss_str + ".pth.tar")
    torch.save(state, filename)
    if is_best_HM_loss:
        print("This model has been saved for its lower heat map loss: ", hm_loss_str)
        torch.save(state, os.path.join(config.ckpt_dir, config.model_name + "-ckpt_LowestHMLoss.pth.tar"))

    if is_best_total_loss:
        print("This model has been saved for its lower total loss:", total_loss_str)
        torch.save(state, os.path.join(config.ckpt_dir, config.model_name + "-ckpt_LowestTotalLoss.pth.tar"))

    if is_best_pose2d_loss:
        print("This model has been saved for its lower 2D pose loss: ", kps_loss_str)
        torch.save(state, os.path.join(config.ckpt_dir, config.model_name + "-ckpt_Lowest2DPoseError.pth.tar"))


def main():
    # Create a directory where checkpoints and training history are saved
    if not os.path.exists(config.ckpt_dir):
        os.mkdir(config.ckpt_dir)
    if not os.path.exists(config.history_dir):
        os.mkdir(config.history_dir)

    torch.cuda.empty_cache()

    logger = logging.getLogger(name='Training log')
    logger.setLevel(logging.INFO)
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logfile = config.root_dir + config.model_name + '_' + rq + '.log'
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到hand
    logger.addHandler(fh)

    # Read datasets
    print("Reading dataset...")
    training_set = hm_dataset(config.root_dir, 'training', input_resize=config.input_size)
    val_set = hm_dataset(config.root_dir, 'evaluation', input_resize=config.input_size)
    train_loader = DataLoader(
        training_set,
        batch_size=config.batch_size,  # 每批样本个数
        shuffle=config.is_shuffle,  # 是否打乱顺序
        drop_last=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,  # 每批样本个数
    )
    config.num_train_samples = len(training_set)
    config.num_val_samples = len(val_set)
    config.num_train_iters = int(len(training_set) / config.batch_size)
    
    print("Found {} training samples and {} validation samples".format(len(training_set), len(val_set)))

    # instantiate a model
    print('Creating a model with {} HourGlass stack(s)'.format(config.stack))
    model = HGFilter(stack=config.stack,
                     depth=config.depth,
                     in_ch=config.in_channels,
                     last_ch=config.last_channels)
    if model is None:
        print("The model created is invalid! Program terminated!")
        exit()
    # recorder.writer.add_graph(model)
    model.to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    resume = False
    start_epoch = 0
    if resume:
        ckpt = torch.load(os.path.join(config.ckpt_dir, config.model_name + "-ckpt_Lowest2DPoseError.pth.tar"))
        start_epoch = ckpt['epoch']
        recorder.val_hm_loss_min = ckpt["val_hm_loss_min"]
        recorder.val_kps_loss_min = ckpt["val_kps_loss_min"]
        recorder.val_total_loss_min = ckpt["val_total_loss_min"]
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma,
                                                       last_epoch=start_epoch)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    # calculate the number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    print("-------------------------Training starts--------------------------")
    start_time = time.time()
    for epoch in range(start_epoch, config.num_epochs):
        recorder.cur_epoch += 1
        epoch_str = "Epoch %d" % (epoch + 1)
        print(epoch_str, "Learning rate: %f   Time: " % (optimizer.param_groups[0]['lr']), end='')
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        epoch_start_time = time.time()

        model.train()
        for i, batch in enumerate(train_loader):
            # batch has three keys:
            #     # key 1: batch['image']         size: B x 3(C) x 256(W) x 256(H)
            #     # key 2: batch['heat_map_gn']   normalized 2D heat maps (each sums to 1) of size: B x 21(kps) x 64(W) x 64(H)
            #     # each of the 21 heat maps is of size 64(W) x 64(H)
            #     # key 3: batch['2d_kp_anno']     size: B x 21(kps) x 2(uv) [u (rightwards),v (downwards)]
            #     # The first row contains u-axis coordinates and the second row contains v-axis values
            input_img = batch['image'].to(device)
            output = model(input_img)  # the output has two items. 1. tmp_outputs: contains outputs from each stage; 2. norm_x

            hm_loss, kps_loss, total_loss = recorder.process_training_output(hm_pred=output[0],
                                                    hm_gn=batch['heat_map_gn'].to(device).float(),
                                                    kps_gn=batch['2d_kp_anno'])

            cur_iter_num = recorder.cur_iter_num + 1
            recorder.cur_iter_num += 1
            if cur_iter_num % config.record_period == 0:
                msg = 'Epoch: [{}][{}/{}]\t' \
                    'Completion Time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\t' \
                    'Heat Map Loss ({:.2e}): {:.4f} \t' \
                    '2D Pose Loss ({:.2e}): {:.4f} \t'  \
                    'Total Loss: {:.4f} \t'.format(
                        epoch+1,
                        cur_iter_num % config.num_train_iters,
                        config.num_train_iters,
                        hm_loss.item(),
                        kps_loss.item(),
                        total_loss.item())

                logger.info(msg)
                recorder.train_hm_loss.append(hm_loss.item())
                recorder.train_kps_loss.append(kps_loss.item())
                recorder.train_total_loss.append(total_loss.item())

                recorder.writer.add_scalar('HM_Loss/Training', hm_loss.item(), cur_iter_num)
                recorder.writer.add_scalar('Pose2D_Loss/Training', kps_loss.item(), cur_iter_num)
                recorder.writer.add_scalar('Total_Loss/Training', total_loss.item(), cur_iter_num)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            del input_img
            del output
            del loss

        print("Evaluating model...")
        is_best_HM_loss, is_best_pose2d_loss, is_best_total_loss = evaluate(val_loader, model)
        print(epoch_str,
              "heat map loss: {:.4f} (The best: {:.4f})".format(recorder.val_hm_loss[-1], recorder.val_hm_loss_min))
        print(epoch_str,
              "2D pose loss: {:.4f} (The best: {:.4f})".format(recorder.val_kps_loss[-1], recorder.val_kps_loss_min))
        print(epoch_str,
              "Total loss: {:.4f} (The best: {:.4f})".format(recorder.val_total_loss[-1], recorder.val_total_loss_min))

        print("Saving model checkpoint...")
        save_checkpoint({
            "epoch": epoch + 1,
            "model_name": config.model_name,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_pose2d_loss": recorder.val_kps_loss[-1],
            "val_hm_loss": recorder.val_hm_loss[-1],
            "val_total_loss": recorder.val_total_loss[-1],
            "val_hm_loss_min": recorder.val_hm_loss_min,
            "val_kps_loss_min": recorder.val_kps_loss_min,
            "val_total_loss_min": recorder.val_total_loss_min
        }, is_best_HM_loss, is_best_pose2d_loss, is_best_total_loss)

        lr_scheduler.step()

        print("Epoch %d spent %d seconds \n" % (epoch + 1, int(time.time() - epoch_start_time)))

    print("{} epochs spent {:.2f} hours".format(config.num_epochs, (time.time()-start_time)/3600))
    recorder.close()


def test():
    class model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
            self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

    initial_lr = 0.01
    net_1 = model()
    optimizer_1 = torch.optim.Adam(net_1.parameters(), lr=initial_lr)
    scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=config.step_size, gamma=config.gamma)

    for epoch in range(10):
        optimizer_1.zero_grad()
        optimizer_1.step()

        print("第%d个epoch的学习率：%f" % (epoch, optimizer_1.param_groups[0]['lr']))
        scheduler_1.step()


def test_hm_loss_fn():
    training_set = hm_dataset(config.root_dir, 'training', input_resize=config.input_size)
    train_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=config.batch_size,  # 每批样本个数
        shuffle=config.is_shuffle,  # 是否打乱顺序
        drop_last=True
    )
    for i, batch in enumerate(train_loader):
        hm_gn = batch['heat_map_gn']
        kps_gn = batch['2d_kp_anno']
        kps_pred = dsnt.spatial_expectation2d(hm_gn, normalized_coordinates=False)
        #kps_pred = spatial_soft_argmax2d(hm_gn, temperature=torch.tensor(10),normalized_coordinates=False)
        print('kps_pred',kps_pred[0][0:5])
        print('kp_pos',kps_gn[0][0:5])
        print(recorder.pose2d_loss_fn(kps_gn,kps_pred))
        input()

    x=torch.zeros((1,3,7,7))
    x[0][0][1][6]=1
    gaussian_blur_kernel = kornia.filters.GaussianBlur2d((7,7),(1,1),border_type='constant')
    y=gaussian_blur_kernel(x)
    for i in range(y.size(1)):
        if y[0][i].sum().item() >0:
            y[0][i] /= y[0][i].sum()

    output: torch.Tensor = dsnt.spatial_expectation2d(y, normalized_coordinates=False)
    print('kps_pred', output[0])
    #print('kp_pos',kps_gn[0][0:5])





if __name__ == '__main__':
    main()
