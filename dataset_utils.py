import os
import shutil
import time

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import kornia
from kornia.geometry import spatial_soft_argmax2d
from kornia.geometry.subpix import dsnt

from HGFilters import HGFilter
from config_file import config
import heatmap_decoding
import metrics

toPIL = transforms.ToPILImage()
toTensor = transforms.ToTensor()

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
        self.images = sorted(os.listdir(self.data_path))
        self.hm_size = hm_size
        self.input_resize = input_resize
        self.anno2d_path = os.path.join(root_dir, mode, 'anno_%s.pickle' % mode)
        with open(self.anno2d_path, 'rb') as fi:
            self.anno_all = pickle.load(fi)  # type: dict

    def __len__(self):
        return len(self.images)

    def generate_heat_maps(self, joint_x, joint_y, visible, hm_size):
        gaussian_blur_kernel = kornia.filters.GaussianBlur2d((config.kernel_size, config.kernel_size),
                                                             (config.sigma, config.sigma),
                                                             border_type='constant')
        hm_lst = []

        for kp_idx in range(21):
            if visible[kp_idx]:
                hm = torch.zeros((1,1,hm_size, hm_size))

                col = min(63,max(0,int(joint_x[kp_idx])))
                row = min(63,max(0,int(joint_y[kp_idx])))

                hm[0][0][row][col] = 1
                normalized_hm = gaussian_blur_kernel(hm).squeeze()
            else:
                normalized_hm = torch.zeros((hm_size, hm_size))
            hm_lst.append(normalized_hm)
        
        return hm_lst

    def generate_heat_maps_without_quantization(self, joint_x, joint_y, visible, hm_size):
        """
        params:
        @ joint_x: u-axis (rightwards)
        @ joint_y: v-axis (downwards)
        """
        hm_lst = []
        sigma = config.sigma

        for kp_idx in range(21):
            if visible[kp_idx]:
                hm = torch.zeros((hm_size, hm_size), dtype=torch.float32)

                col = min(63.0,max(0.0, joint_x[kp_idx]))
                row = min(63.0,max(0.0, joint_y[kp_idx]))
                row_idx, col_idx = 0, 0
                for i in range(0, 6 * sigma + 1):
                    row_idx = int(row) - 3 * sigma + i
                    if 0 <= row_idx < hm_size:
                        for j in range(0, 6 * sigma + 1):
                            col_idx = int(col) - 3 * sigma + j
                            if 0 <= col_idx < hm_size:
                                d = (row_idx - row) ** 2 + (col_idx - col) ** 2
                                if d < 16 * sigma * sigma:
                                    hm[row_idx][col_idx] = torch.exp(-1 * torch.tensor(d, dtype=torch.float32) / (2 * sigma * sigma))
                                    
                normalized_hm = hm / hm.sum()
            else: 
                normalized_hm = torch.zeros((hm_size, hm_size))

            hm_lst.append(normalized_hm)
        return hm_lst


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
        hm_lst = self.generate_heat_maps_without_quantization(one_hand_kp_x_for_cropping,
                                    one_hand_kp_y_for_cropping,
                                    one_hand_visible_kps,
                                    hm_size)
        
        # print("l:{} r:{} b:{} t:{} size:{}".format(leftmost, rightmost, bottommost, topmost, crop_size))
        # print("top_left corner:", top_left_corner)
        # print("crop size:", cropped_img.size)

        # cropped_img_tensor = toTensor(cropped_img)  # [3, W, H]
        #
        # for i in range(0,21,5):
        #     fig = plt.figure(1)
        #     ax1 = fig.add_subplot(1,3,1)
        #     ax2 = fig.add_subplot(1,3,2)
        #     ax3 = fig.add_subplot(1,3,3)
        
        #     ax1.imshow(img)  # the original image
        #     ax1.plot(one_hand_kp_x[i], one_hand_kp_y[i], 'r*')
        #     ax1.plot([top_left_corner[0], top_left_corner[0] + crop_size],
        #              [top_left_corner[1], top_left_corner[1] + crop_size], 'r')  # the cropped area
        
        #     cropped_img_tensor = cropped_img.resize((self.hm_size, self.hm_size))
        #     ax2.imshow(cropped_img_tensor)
        #     ax2.plot(one_hand_kp_x_for_cropping[i], one_hand_kp_y_for_cropping[i], 'r*')
        #     print()
        #     ax3.imshow(toPIL(255 * hm_lst[i]))
        #     plt.show()


        transform_for_img = transforms.Compose([transforms.Resize((self.input_resize, self.input_resize)),
                                                transforms.ToTensor()
                                                ])
        one_sample = {'image': transform_for_img(cropped_img),  ## type: torch.Tensor
                      'heat_map_gn': torch.stack(hm_lst),  ## type: torch.tensor size: [21,w,h]
                      '2d_kp_anno': np.vstack((one_hand_kp_x_for_cropping, one_hand_kp_y_for_cropping)).transpose()
                      ## type: np.array size: 21 x 2
                      }

        return one_sample

def visualize_samples(data_loader):
    # visualize some samples
    for batch in data_loader:
        # batch has three keys:
        # key 1: batch['image']         size: batch_size x 3(C) x 256(W) x 256(H)
        # key 2: batch['heat_map_gn']   size: batch_size x 21(kps) x 64(W) x 64(H)
        # each of the 21 heat maps is of size 64(W) x 64(H)
        # key 3: batch['2d_kp_anno']     size: batch_size x 21(kps) x 2 [u,v]
        # The first row contains u-axis coordinates and the second row contains v-axis values
        kps = batch['2d_kp_anno'][0]
        hms = batch['heat_map_gn']
        print(kps)
        for i in range(0, batch['heat_map_gn'].shape[1], 6):
            fig = plt.figure(1)
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)
            img = toPIL(batch['image'][0]).resize((config.heat_map_size, config.heat_map_size))
            ax1.imshow(img)
            ax1.plot(kps[i][0],kps[i][1],'r*')
            ax2.imshow(toPIL(hms[0][i]))
            
            kps_pred, _ = heatmap_decoding.get_final_preds(hms.cpu().numpy())
            #kps_pred =  dsnt.spatial_expectation2d(hms[i].view(1,1,config.heat_map_size,config.heat_map_size), normalized_coordinates=False)
            print("Gn:",kps[i],"max idx: [{}, {}]".format(hms[0][i].argmax().item() % config.heat_map_size, hms[0][i].argmax().item() // config.heat_map_size),"\t pred:",kps_pred[0][i])
            plt.show()


if __name__ == '__main__':
    val_set = hm_dataset(config.root_dir, 'training', input_resize=config.input_size)
    val_dataloader = DataLoader(
        val_set,
        batch_size=1,  # 每批样本个数
    )
    visualize_samples(val_dataloader)
