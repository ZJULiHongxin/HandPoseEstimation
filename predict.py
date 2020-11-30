import os
import shutil
import time
import argparse

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
from net_util import conv3x3
from config_file import config
import metrics
from dataset_utils import hm_dataset


def assess_model():
    img_dir = 'evaluation/color'
    transform_for_img = transforms.Compose([transforms.Resize((config.heat_map_size, config.heat_map_size)),
                                                transforms.ToPILImage()
                                                ])
    toPIL = transforms.ToPILImage()
    recorder = metrics.MetricsRecorder()
    device = metrics.device

    legend_lst = ['wrist',
         'thumb tip', 'thumb near tip','thumb near palm', 'palm',
        'index tip', 'index near tip', 'index near palm', 'index palm',
        'middle tip', 'middle near tip', 'middle near palm', 'middle palm',
        'ring tip', 'ring near tip', 'ring near palm', 'ring palm',
        'pinky tip', 'pinky near tip', 'pinky near palm', 'pinky palm']
    
    val_set = hm_dataset(config.root_dir, 'training', input_resize=config.input_size)
    val_dataloader = DataLoader(
        val_set,
        batch_size=1,  # 每批样本个数
    )

    model = HGFilter(stack=config.stack,
                     depth=config.depth,
                     in_ch=config.in_channels,
                     last_ch=config.last_channels)
    model.to(device)

    ckpt = torch.load(os.path.join(config.ckpt_dir, config.model_name + "-ckpt_Lowest2DPoseError.pth.tar"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            # batch has three keys:
            # key 1: batch['image']         size: batch_size x 3(C) x 256(W) x 256(H)
            # key 2: batch['heat_map_gn']   size: batch_size x 21(kps) x 64(W) x 64(H)
            # each of the 21 heat maps is of size 64(W) x 64(H)
            # key 3: batch['2d_kp_anno']     size: batch_size x 21(kps) x 2 [u,v]
            # The first row contains u-axis coordinates and the second row contains v-axis values
            img = batch['image'].to(device)
            hm_gn = batch['heat_map_gn']
            kps_gn = batch['2d_kp_anno'].to(device)

            start_time = time.time()
            output = model(img)

            print('Inference time: {:.4f} s'.format(time.time()-start_time))
            
            kps_pred, EPE = recorder.assess_performance(hm_pred=output[0],
                                               kps_gn=kps_gn)
            kps_gn = kps_gn.squeeze().cpu().numpy()
            kps_pred = kps_pred.squeeze().cpu().numpy()
            print('Ground Truth')
            print(kps_gn)
            print('End point error:')
            for i in range(21):
                print(legend_lst[i]+(25-len(legend_lst[i]))*' '+'location: [{:.2f}, {:.2f}]  error: {:.4f}'.format(kps_pred[i][0], kps_pred[i][1], EPE[i].item()))
            
            print('Avg: {:.4f}'.format(EPE.mean().item()))

            hms = output[0][-1]
            img = transform_for_img(img.squeeze())
            for kp in range(0,21,5):
                fig = plt.figure(1)
                ax1 = fig.add_subplot(2,2,1)
                ax2 = fig.add_subplot(2,2,2)
                ax3 = fig.add_subplot(2,2,3)
                ax4 = fig.add_subplot(2,2,4)
                # real key point location
                ax1.imshow(img)
                ax1.plot(kps_gn[kp,0], kps_gn[kp,1], 'r*')
                # predicted key point location
                ax2.imshow(img)
                ax2.plot(kps_pred[kp,0], kps_pred[kp,1], 'r*')
                # real key point heat map
                ax3.imshow(toPIL(hm_gn[0][kp]))
                print(hm_gn[0][kp].sum())
                # predicted key point heat map
                hm = 255 * hms[0][kp] / hms[0][kp].sum()
                ax4.imshow(toPIL(hm))
                print(hm.sum())

                plt.show()

    

    


if __name__ == '__main__':
    assess_model()