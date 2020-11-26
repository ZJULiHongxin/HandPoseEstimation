import os
import shutil
import time

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle
import torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader

from HGFilters import HGFilter
from net_util import conv3x3
from config_file import config
import metrics

toPIL = transforms.ToPILImage()
toTensor = transforms.ToTensor()
recorder = metrics.MetricsRecorder()
device = metrics.device

class hm_dataset(Dataset):
    def __init__(self, root_dir, mode, input_resize=256, hm_size=64):
        r"""Read RGB images and their ground truth heat maps
        params:
        @ root_dir: the directory where the training and test datasets are located
        @ mode: indicates whether the this is a training set ot a test set
        @ transform: data augmentation is implemented if transformers are given
        """

        self.root_dir = root_dir
        self.data_path = os.path.join(root_dir, mode, 'color')
        self.images = os.listdir(self.data_path)
        self.hm_size = hm_size
        self.input_resize = input_resize
        self.anno2d_path = os.path.join(root_dir, mode, 'anno_%s.pickle' % mode)
        with open(self.anno2d_path, 'rb') as fi:
            self.anno_all = pickle.load(fi)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        hm_size = self.hm_size
        # read an image
        img_name = self.images[index]
        img_path = os.path.join(self.data_path, img_name)
        img = Image.open(img_path)

        # read the 2D keypoints annotation of this image. Note: "self.anno_all[index]['uv_vis']"" is a numpy array of size 42x3.
        # The first colum of 2D kp annotation matrix represents the x-axis (horizontal and positive rightwards) value of a key point;
        # the second column represents the y-axis (vertical and positive upwards) value of a key point;
        # the third column consists boolean values donoting the visibility of key points (1 for visible points and 0 otherwise)
        kp_coord_uv = self.anno_all[index]['uv_vis'][:, :2]  # u, v coordinates of 42 hand key points, pixel
        kp_visible = (self.anno_all[index]['uv_vis'][:, 2] == 1)  # visibility of the key points, boolean

        num_left_hand_visible_kps = np.sum(self.anno_all[index]['uv_vis'][0:21, 2])
        num_right_hand_visible_kps = np.sum(self.anno_all[index]['uv_vis'][21:42, 2])

        # crop the image so that it contains only one hand which has the most visible key points
        if num_left_hand_visible_kps > num_right_hand_visible_kps:
            one_hand_kp_x = kp_coord_uv[0:21, 0].copy()
            one_hand_kp_y = kp_coord_uv[0:21, 1].copy()
            one_hand_visible_kps = kp_visible[0:21]
        else:
            one_hand_kp_x = kp_coord_uv[21:42, 0].copy()
            one_hand_kp_y = kp_coord_uv[21:42, 1].copy()
            one_hand_visible_kps = kp_visible[21:42]

        leftmost, rightmost = np.min(one_hand_kp_x), np.max(one_hand_kp_x)
        bottommost, topmost = np.max(one_hand_kp_y), np.min(one_hand_kp_y)
        w, h = rightmost - leftmost, bottommost - topmost

        crop_size = min(img.size[0], int(2 * w if w > h else 2 * h))

        # top_left_corner of the cropped area: [u, v] in u-v image system ↓→
        top_left_corner = (max(0, min(int(leftmost - (crop_size - w) / 2), img.size[0] - crop_size)),
                           max(0, min(img.size[1] - crop_size, int(topmost - (crop_size - h) / 2))))

        cropped_img = img.crop(  # a PIL image
            (
                top_left_corner[0],  # The distance of the left border to the left border of the original image
                top_left_corner[1],  # The distance of the top border to the top border of the original image
                top_left_corner[0] + crop_size,
                # The distance of the right border to the left border of the original image
                top_left_corner[1] + crop_size
                # The distance of the bottom border to the top border of the original image
            )
        )

        # calculate the ground truth positions of key points on heat maps
        scale_factor = hm_size / crop_size
        one_hand_kp_x_for_cropping = scale_factor * (one_hand_kp_x - top_left_corner[0])
        one_hand_kp_y_for_cropping = scale_factor * (one_hand_kp_y - top_left_corner[1])

        # create heat map ground truth from key points
        # 1. create a Gaussian probability distribution
        sigma = config.sigma  # Gaussian blur kernel size
        patch = np.zeros(shape=[6 * sigma + 1, 6 * sigma + 1],
                         dtype=np.float)
        patch[3 * sigma, 3 * sigma] = 255
        patch = cv2.GaussianBlur(patch, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)

        # 2. generate a heat map for each visible key point
        # For ground truth generation of the score maps, we use normal distributions with a standard deviation of 2 pixels and
        # the mean being equal to the given key point location. We normalize the resulting maps such that
        # each map contains values from 0 to 1, if there is a keypoint visible.
        # For invisible keypoints the map is zero everywhere.
        hm_lst = []

        for kp_idx in range(21):
            if one_hand_visible_kps[kp_idx]:
                hm = np.zeros((hm_size, hm_size), dtype=np.float)

                col = int(one_hand_kp_x_for_cropping[kp_idx])
                row = int(one_hand_kp_y_for_cropping[kp_idx])

                # add the patch onto the black background to get a heat map
                row_idx, col_idx = 0, 0
                for i in range(0, 6 * sigma + 1):
                    row_idx = row - 3 * sigma + i
                    if 0 <= row_idx < hm_size:
                        for j in range(0, 6 * sigma + 1):
                            col_idx = col - 3 * sigma + j
                            if 0 <= col_idx < hm_size:
                                hm[row_idx, col_idx] = patch[i, j]
                normalized_hm = hm / hm.sum(0).sum(0)

            else:
                normalized_hm = np.zeros((self.hm_size, self.hm_size), dtype=np.float)
            hm_lst.append(normalized_hm)

        # print("l:{} r:{} b:{} t:{} size:{}".format(leftmost, rightmost, bottommost, topmost, crop_size))
        # print("top_left corner:", top_left_corner)
        # print("crop size:", cropped_img.size)

        # cropped_img_tensor = toTensor(cropped_img)  # [3, W, H]
        #
        # for i in range(0,21,5):
        #     fig = plt.figure(1)
        #     ax1 = fig.add_subplot('131')
        #     ax2 = fig.add_subplot('132')
        #     ax3 = fig.add_subplot('133')
        #
        #     ax1.imshow(img)  # the original image
        #     ax1.plot(one_hand_kp_x[i], one_hand_kp_y[i], 'r*')
        #     ax1.plot([top_left_corner[0], top_left_corner[0] + crop_size],
        #              [top_left_corner[1], top_left_corner[1] + crop_size], 'r')  # the cropped area
        #
        #     cropped_img_tensor = cropped_img.resize((self.hm_size, self.hm_size))
        #     ax2.imshow(cropped_img_tensor)
        #     ax2.plot(one_hand_kp_x_for_cropping[i], one_hand_kp_y_for_cropping[i], 'r*')
        #
        #     heat_map_uint8 = (255 * hm_lst[i]).astype('uint8')
        #     heat_map_RGB = toTensor(heat_map_uint8)  # [3, W, H]
        #     ax3.imshow(toPIL(heat_map_RGB))
        #
        #     plt.show()

        transform_for_img = transforms.Compose([transforms.Resize((self.input_resize, self.input_resize)),
                                                transforms.ToTensor()
                                                ])
        one_sample = {'image': transform_for_img(cropped_img),  ## type: torch.Tensor
                      'heat_map_gn': np.stack(hm_lst, axis=0),  ## type: np.array size: [21,w,h]
                      '2d_kp_anno': np.vstack((one_hand_kp_x_for_cropping, one_hand_kp_y_for_cropping)).transpose()
                      ## type: np.array size: 21 x 2
                      }

        return one_sample


def visualize_samples(data_loader):
    # visualize some samples
    for batch in data_loader:
        # batch has three keys:
        # key 1: batch['image']         size: batch_size x 3(C) x 192(W) x 192(H)
        # key 2: batch['heat_map_gn']   size: batch_size x 21(kps) x 64(W) x 64(H)
        # each of the 21 heat maps is of size 64(W) x 64(H)
        # key 3: batch[2d_kp_anno']     size: batch_size x 2 x 21(kps)
        # The first row contains u-axis coordinates and the second row contains v-axis values

        for i in range(0, batch['heat_map_gn'].shape[1], 6):
            fig = plt.figure(1)
            ax1 = fig.add_subplot('121')
            ax2 = fig.add_subplot('122')
            img = toPIL(batch['image'][0])
            ax1.imshow(img)
            ax2.imshow(toPIL(batch['heat_map_gn'][0][i]))
            print(torch.max(batch['heat_map_gn'][0][i]))
            plt.show()


def evaluate(val_data_loader, model):
    model.eval()  # model.eval()会仍会执行gradient的计算和存储，只是不进行反传
    recorder.eval()
    with torch.no_grad():  # 停止autograd模块的工作，以起到加速和节省显存的作用，但是并不会影响dropout和batchnorm层的行为
        for i, batch in enumerate(val_data_loader):
            input_img = batch['image'].to(device)  # detach: no-gradient
            hm_groundtruth = batch['heat_map_gn'].to(device)
            output = model(input_img)
            kps_gn = batch['2d_kp_anno'].to(device)
            recorder.process_validation_output(hm_pred=output[0][-1],
                                               hm_gn=hm_groundtruth,
                                               kps_gn=kps_gn)
            del input_img
            del hm_groundtruth
            del output
    return recorder.finish_eval()


def save_checkpoint(state: dict, is_best_HM_loss, is_best_EPE, is_best_PCK):
    hm_loss_str = "{:.4f}".format(recorder.val_hm_loss[-1])
    EPE_str = "{:.4f}".format(recorder.val_avg_EPE[-1])
    PCK_str = "{:.4f}".format(recorder.val_avg_PCK[-1])

    filename = os.path.join(config.ckpt_dir,
                            config.model_name + "-ckpt_HM_Loss_" + hm_loss_str + "_EPE_" + EPE_str + "_PCK_" + PCK_str + ".pth.tar")
    torch.save(state, filename)
    if is_best_HM_loss:
        print("This model has been saved for its lower heat map loss: ", hm_loss_str)
        torch.save(state, os.path.join(config.ckpt_dir, config.model_name + "-ckpt_LowestHMLoss.pth.tar"))

    if is_best_PCK:
        print("This model has been saved for its higher PCK:", PCK_str)
        torch.save(state, os.path.join(config.ckpt_dir, config.model_name + "-ckpt_HighestPCK.pth.tar"))

    if is_best_EPE:
        print("This model has been saved for its lower EPE: ", EPE_str)
        torch.save(state, os.path.join(config.ckpt_dir, config.model_name + "-ckpt_Lowest2DPoseError.pth.tar"))


def main():
    # Create a directory where checkpoints and training history are saved
    if not os.path.exists(config.ckpt_dir):
        os.mkdir(config.ckpt_dir)
    if not os.path.exists(config.history_dir):
        os.mkdir(config.history_dir)

    torch.cuda.empty_cache()

    # Read datasets
    print("Reading dataset...")
    training_set = hm_dataset(config.root_dir, 'training', input_resize=config.input_size)
    val_set = hm_dataset(config.root_dir, 'evaluation', input_resize=config.input_size)
    train_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=config.batch_size,  # 每批样本个数
        shuffle=config.is_shuffle,  # 是否打乱顺序
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config.batch_size,  # 每批样本个数
        shuffle=config.is_shuffle,  # 是否打乱顺序
    )
    config.num_train_samples = len(training_set)
    config.num_val_samples = len(val_set)
    config.num_iters = int(len(training_set)/config.batch_size)
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
    #recorder.writer.add_graph(model)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    resume = False
    start_epoch = 0
    if resume:
        ckpt = torch.load(os.path.join(config.ckpt_dir, config.model_name + "-ckpt_Lowest2DPoseError.pth.tar"))
        start_epoch = ckpt['epoch']
        recorder.val_hm_loss_min = ckpt["valid_hm_loss"]
        recorder.val_EPE_min = ckpt["EPE"]
        recorder.val_PCK_max = ckpt["PCK"]
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
    for epoch in range(start_epoch, config.num_epochs):
        epoch_str = "Epoch %d" % (epoch+1)
        print(epoch_str, "Learning rate: %f   Time: " % (optimizer.param_groups[0]['lr']), end='')
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        start_time = time.time()
        for i, batch in enumerate(train_loader):
            # batch has three keys:
            #     # key 1: batch['image']         size: B x 3(C) x 192(W) x 192(H)
            #     # key 2: batch['heat_map_gn']   size: B x 21(kps) x 64(W) x 64(H)
            #     # each of the 21 heat maps is of size 64(W) x 64(H)
            #     # key 3: batch['2d_kp_anno']     size: B x 21(kps) x 2(uv)
            #     # The first row contains u-axis coordinates and the second row contains v-axis values

            model.train()
            input_img = batch['image'].to(device)
            hm_groundtruth = batch['heat_map_gn'].to(device)
            # the output has two items. 1. tmp_outputs: contains outputs from each stage
            # 2. norm_x
            output = model(input_img)
            loss = recorder.process_training_output(hm_pred=output[0][-1],
                                                    hm_gn=hm_groundtruth,
                                                    kps_gn=batch['2d_kp_anno'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del input_img
            del output
            del loss

        print("Evaluating model...")
        is_best_HM_loss, is_best_EPE, is_best_PCK = evaluate(val_loader, model)
        print(epoch_str, "heat map loss: {:.4f} (The best: {:.4f})".format(recorder.val_hm_loss[-1], recorder.val_hm_loss_min))
        print(epoch_str, "2D pose loss: {:.4f} (The best: {:.4f})".format(recorder.val_avg_EPE[-1], recorder.val_EPE_min))
        print(epoch_str, "PCK: {:.4f} (The best: {:.4f})".format(recorder.val_avg_PCK[-1], recorder.val_PCK_max))

        print("Saving model checkpoint...")
        save_checkpoint({
            "epoch": epoch + 1,
            "model_name": config.model_name,
            "state_dict": model.state_dict(),
            "EPE": recorder.val_avg_EPE[-1],
            "PCK": recorder.val_avg_PCK[-1],
            "optimizer": optimizer.state_dict(),
            "valid_hm_loss": recorder.val_hm_loss[-1],
        }, is_best_HM_loss, is_best_EPE, is_best_PCK)

        lr_scheduler.step()

        print("Epoch %d spent %d seconds \n" % (epoch+1, int(time.time()-start_time)))

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


if __name__ == '__main__':
    main()
