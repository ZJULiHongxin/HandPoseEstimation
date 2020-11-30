import os
import time
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader

import kornia
from kornia.geometry import spatial_soft_argmax2d
from kornia.geometry.subpix import dsnt

from HGFilters import HGFilter
from config_file import config
from dataset_utils import hm_dataset
import metrics


def assess_model():
    recorder = metrics.MetricsRecorder()
    device = metrics.device
    val_set = hm_dataset(config.root_dir, 'evaluation', input_resize=config.input_size)
    val_dataloader = DataLoader(
        val_set,
        batch_size=1,  # 每批样本个数
    )
    config.num_val_samples = len(val_set)
    model = HGFilter(stack=config.stack,
                     depth=config.depth,
                     in_ch=config.in_channels,
                     last_ch=config.last_channels)
    model.to(device)

    ckpt = torch.load(os.path.join(config.ckpt_dir, config.model_name + "-ckpt_Lowest2DPoseError.pth.tar"))
    model.load_state_dict(ckpt["state_dict"])

    
    model.eval() 
    print('Start assessing...')
    start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            input_img = batch['image'].to(device)  # detach: no-gradient
            output = model(input_img)
            kps_gn = batch['2d_kp_anno'].to(device)
            recorder.assess_performance(hm_pred=output[0],
                                               kps_gn=kps_gn)
            del input_img
            del output
        recorder.finish_assessment()
    print('Assessment completed. Time elapsed: {:d} s.'.format(int(time.time()-start_time)))


if __name__ == '__main__':
    assess_model()