import os
import torch

# from torch.nn.parameter import Parameter
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import kornia
from kornia.geometry.subpix import dsnt
import matplotlib.pyplot as plt

import heat_map_utils
from config_file import config

# Check the availability of GPUs
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Uncomment this to run on GPU
    print("\nUsing one GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')


def save_list(file_name, lst):
    with open(file_name, 'w') as f:
        s = str(lst)
        s = s.replace('[', '').replace(']', '').replace(',', '')
        f.write(s)


def save_multilist(file_name, multilst):
    with open(file_name, 'w') as f:
        for lst in multilst:
            s = str(lst)
            s = s.replace('[', '').replace(']', '').replace(',', '') + '\n'
            f.write(s)


class MetricsRecorder(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    __slots__ = "hm_loss_fn", "calc_kps_with_softmax", "pose2d_loss_fn", "cur_iter_num", "cur_epoch",\
                "num_val_iters", "writer", "val_hm_loss", "test_PCK_of_each_joint", "test_EPE_of_each_joint", \
                "val_hm_loss_min", "val_kps_loss_min", "new_val_flag", "cur_iter_num", "accumulated_loss", "val_total_loss",\
                "val_PCK_max", "train_hm_loss", "train_kps_loss", "train_total_loss", "val_avg_PCK", "val_kps_loss",\
                "val_total_loss_min", "val_avg_PCK_over_all_joints", "val_avg_EPE_over_all_joints", "th_lst"

    def __init__(self):
        self.hm_loss_fn = torch.nn.MSELoss(reduction='sum')  # Mean-square error: sum((A-B).^2)                                             temperature=config.temperature).to(device)
        self.pose2d_loss_fn = torch.nn.MSELoss(reduction='sum')  # Mean-square error: sum((A-B).^2)
        self.cur_epoch = 0
        self.cur_iter_num = 0
        self.num_val_iters = 0
        self.writer = SummaryWriter()

        self.val_hm_loss = []
        self.val_total_loss = []
        self.val_kps_loss = []  # unit: px
        self.val_hm_loss_min = 1e6
        self.val_total_loss_min = 1e6
        self.val_kps_loss_min = 1e6

        self.train_hm_loss = []
        self.train_kps_loss = []
        self.train_total_loss = []

        self.th_lst = [i for i in range(20)]
        self.test_PCK_of_each_joint = torch.zeros((21, len(self.th_lst))).to(device)  # The percentages of correct key points
        self.test_EPE_of_each_joint = torch.zeros((21, 1)).to(device)  # The end-point errors of correct key points
        self.val_avg_PCK_over_all_joints = []
        self.val_avg_EPE_over_all_joints = 0
        

    """
    These four functions are used in training.py
    """
    def process_training_output(self, hm_pred: torch.Tensor, hm_gn: torch.Tensor, kps_gn: torch.Tensor):
        """
        This function computes all loss functions and record current performance.

        params:
        @ hm_pred: 21 predicted heat maps of size stage x B x 21(kps) x 64(W) x 64(H)
        @ hm_gn: The ground truth heat maps of size B x 21(kps) x 64(W) x 64(H)
        @ kps_gn: The 2D key point annotations of size B x K x 2

        retval:
        @ toal_loss: 
        """

        # 1. Multi-stage heat map loss: the MSE - sum((A-B).^2)/N
        # normalize heat maps to make each one sum to 1

        hm_loss = torch.tensor(0,dtype=torch.float).to(device)
        for stage in range(len(hm_pred)):
            hm_loss += self.hm_loss_fn(hm_pred[stage], hm_gn) / (config.batch_size * 21)  # size: a scalar

        # 2. 2D pose loss: the average distance between predicted key points and the ground truth
        # heat map decoding method 1: arg softmax
        # kps_pred = kornia.geometry.spatial_soft_argmax2d(hm_pred)  # size: (B*21) x 2

        # heat map decoding method 1: simple spatial weighted sum
        # hm_pred_final = hm_pred[-1]
        # kps_pred = dsnt.spatial_expectation2d(hm_pred_final, normalized_coordinates=False).view(-1,2).to(device) # size: B x 21 x 2

        # kps_gn = kps_gn.view(-1, 2).to(device)  # change its size to (B*21) x 2
        # kps_loss = self.pose2d_loss_fn(kps_pred, kps_gn) / (config.batch_size * 21)  # loss function: Euclidean distance, size:
        kps_loss = torch.tensor(0)
        total_loss = config.hm_loss_weight * hm_loss # + config.kps_loss_weight * kps_loss

        return hm_loss, kps_loss, total_loss

    def eval(self):
        self.val_hm_loss.append(0)
        self.val_kps_loss.append(0)
        self.val_total_loss.append(0)

    def finish_eval(self):
        is_best_HM_loss, is_best_pose2d_loss, is_best_total_loss = False, False, False

        self.val_hm_loss[-1] /= config.num_val_samples
        self.val_kps_loss[-1] /= config.num_val_samples
        self.val_total_loss[-1] /= config.num_val_samples

        self.num_val_iters += 1
        self.writer.add_scalar('HM_Loss/Validation', self.val_hm_loss[-1], self.num_val_iters)
        self.writer.add_scalar('Pose2D_Loss/Validation', self.val_kps_loss[-1], self.num_val_iters)
        self.writer.add_scalar('Total_Loss/Validation', self.val_total_loss[-1], self.cur_iter_num)
        
        if self.val_kps_loss_min > self.val_kps_loss[-1]:
            self.val_kps_loss_min = self.val_kps_loss[-1]
            is_best_pose2d_loss = True

        if self.val_hm_loss_min > self.val_hm_loss[-1]:
            self.val_hm_loss_min = self.val_hm_loss[-1]
            is_best_HM_loss = True

        if self.val_total_loss_min > self.val_total_loss[-1]:
            self.val_total_loss_min = self.val_total_loss[-1]
            is_best_total_loss = True

        return is_best_HM_loss, is_best_pose2d_loss, is_best_total_loss

    def process_validation_output(self, hm_pred: torch.Tensor, hm_gn: torch.Tensor, kps_gn: torch.Tensor):
        """
        This function computes all loss functions and record current performance.

        params:
        @ hm_pred: 21 predicted heat maps of size B x 21(kps) x 64(W) x 64(H)
        @ hm_gn: The ground truth heat maps of size B x 21(kps) x 64(W) x 64(H)
        @ kps_gn: The 2D key point annotations of size B x K x 2

        retval:
        @ flag: True means the current model is better.
        """
        hm_loss = torch.tensor(0,dtype=torch.float).to(device)
        for stage in range(len(hm_pred)):
            hm_loss += self.hm_loss_fn(hm_pred[stage], hm_gn) / 21

        hm_pred_final = hm_pred[-1]

        self.val_hm_loss[-1] += hm_loss.item()

        # heat map decoding
        kps_pred = dsnt.spatial_expectation2d(hm_pred_final, normalized_coordinates=False).view(-1,2)
        kps_gn = kps_gn.view(-1, 2).to(device)  # change its size to 21 x 2
        kps_loss = self.pose2d_loss_fn(kps_pred, kps_gn) / 21

        self.val_kps_loss[-1] += kps_loss.item()
        self.val_total_loss[-1] += hm_loss.item() + kps_loss.item()

    """
    These 
    """
    # def start_testing(self):
    #     """
    #     record 
    #     """
    #     for i in range(21):
    #         self.test_PCK_of_each_joint[i].append(0) 
    #     self.val_avg_PCK_over_all_joints.append(0)
    def assess_performance(self, hm_pred: torch.Tensor, kps_gn: torch.Tensor):
        hm_size = config.heat_map_size
        hm_pred_final = hm_pred[-1] # size 1 x 21 x W x H
        for i in range(21):
            hm_pred_final[0][i] /= hm_pred_final[0][i].sum()
        
        kps_pred = dsnt.spatial_expectation2d(hm_pred_final, normalized_coordinates=False).view(-1,2)
        kps_gn = kps_gn.view(-1, 2).to(device)  # change its size to 21 x 2

        Euclidean_dist = torch.norm(kps_pred - kps_gn, dim=1, keepdim=True)  # size: 21 x 1
        
        # record PCK
        for i in range(len(self.th_lst)):
            PCK_each_joint = (Euclidean_dist < self.th_lst[i]).float()
            for kp in range(21):
                self.test_PCK_of_each_joint[kp][i] += PCK_each_joint[kp][0]

        # record EPE
        self.test_EPE_of_each_joint += Euclidean_dist

        return kps_pred, Euclidean_dist
            
    def finish_assessment(self):
        PCK_AUC_of_each_joint = self.test_PCK_of_each_joint / config.num_val_samples
        EPE_of_each_joint = self.test_EPE_of_each_joint / config.num_val_samples
        PCK_AUC_over_all_joints = PCK_AUC_of_each_joint.sum(dim=0) / 21
        EPE_over_all_joints = EPE_of_each_joint.sum(dim=0) / 21
        
        with open('./PCK_AUC.txt','w') as f:
            for kp in range(21):
                for th in range(len(self.th_lst)):
                    f.write(str(PCK_AUC_of_each_joint[kp][th].item())+' ')
                f.write('\n')
            
            for th in range(len(self.th_lst)):
                f.write(str(PCK_AUC_over_all_joints[th].item())+' ')
        
        with open('./EPE.txt','w') as f:
            for kp in range(21):
                f.write(str(EPE_of_each_joint[kp])+' ')
        
        # plot performance assessment results
        plt.figure(1)
        plt.subplot(1,2,1)
        plt.plot(self.th_lst, PCK_AUC_over_all_joints.cpu().numpy(), marker='o')
        plt.xlabel('threshold [px]')
        plt.ylabel('PCK')
        plt.title('PCK AUC over all joints')

        plt.subplot(1,2,2)
        legend_lst = ['wrist',
         'thumb tip', 'thumb near tip','thumb near palm', 'palm',
        'index tip', 'index near tip', 'index near palm', 'index palm',
        'middle tip', 'middle near tip', 'middle near palm', 'middle palm',
        'ring tip', 'ring near tip', 'ring near palm', 'ring palm',
        'pinky tip', 'pinky near tip', 'pinky near palm', 'pinky palm']
        
        for kp in range(21):
            plt.plot(self.th_lst, PCK_AUC_of_each_joint[kp].cpu().numpy())
        plt.legend(legend_lst)
        plt.xlabel('threshold [px]')
        plt.ylabel('PCK')
        plt.title('PCK AUC of each joint')

        plt.show()

    def calc_PCK_EPE(self, hm_pred: torch.tensor, kps_gn: torch.tensor, th: int):
        """
        This function calculates PCK and EPE for each joint but returns average PCK and EPE over all joints.

        params:
        @ hm_pred: 21 predicted heat maps of size B x K(21 kps) x 64(W) x 64(H)
        @ kps_gn: The 2D key point annotations of size B x K x 2
        @ th: The threshold within which a predicted key point is seen as correct

        usage:
        

        """
        # find the predicted key points with the most probability. Size: B x K x 3
        pred_kp = heat_map_utils.compute_uv_from_heatmaps(hm=hm_pred, resize_dim=config.heat_map_size)
        pred_kp = pred_kp.to(device)
        # calculate end-point error
        EPE_each_batch_each_joint = torch.linalg.norm(pred_kp[:, :, 0:2] - kps_gn, dim=2)  # size: B x K
        PCK_each_batch_each_joint = EPE_each_batch_each_joint < th  # size: B x K

        del pred_kp
        return EPE_each_batch_each_joint, PCK_each_batch_each_joint

    def close(self):
        self.writer.flush()
        self.writer.close()

        save_list(os.path.join(config.history_dir, config.model_name + '_train_hm_loss.txt'), self.train_hm_loss)
        save_list(os.path.join(config.history_dir, config.model_name + '_train_kps_loss.txt'), self.train_kps_loss)
        save_list(os.path.join(config.history_dir, config.model_name + '_train_total_loss.txt'), self.train_total_loss)
        save_list(os.path.join(config.history_dir, config.model_name + '_val_hm_loss.txt'), self.val_hm_loss)
        save_list(os.path.join(config.history_dir, config.model_name + '_val_kps_loss.txt'), self.val_kps_loss)
        save_list(os.path.join(config.history_dir, config.model_name + '_val_total_loss.txt'), self.val_total_loss)
        


if __name__=='__main__':
   pass
