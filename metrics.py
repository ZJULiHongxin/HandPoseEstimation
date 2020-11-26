import os
import torch
# from torch.nn.parameter import Parameter
from torch.utils.tensorboard import SummaryWriter
from kornia.geometry.subpix import dsnt

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
    __slots__ = "hm_loss_fn", "calc_kps_with_softmax", "pose2d_loss_fn", "num_train_iters", \
                "num_val_iters", "writer", "val_hm_loss", "val_PCK_of_each_joint", "val_EPE_of_each_joint", \
                "val_hm_loss_min", "val_kps_loss_min", "new_val_flag", "count", "accumulated_loss", "val_total_loss",\
                "val_PCK_max", "train_hm_loss", "train_kps_loss", "train_total_loss", "val_avg_PCK", "val_kps_loss",\
                "val_total_loss_min"

    def __init__(self):
        self.hm_loss_fn = torch.nn.MSELoss(reduction='sum')  # Mean-square error: sum((A-B).^2)                                             temperature=config.temperature).to(device)
        self.pose2d_loss_fn = torch.nn.MSELoss(reduction='sum')  # Mean-square error: sum((A-B).^2)
        self.num_train_iters = 0
        self.num_val_iters = 0
        self.writer = SummaryWriter()

        self.val_hm_loss = []
        self.val_total_loss = []
        self.val_PCK_of_each_joint = [[] for _ in range(21)]  # The percentages of correct key points
        self.val_EPE_of_each_joint = [[] for _ in range(21)]  # The end-point errors of correct key points
        self.val_avg_PCK = []
        self.val_kps_loss = []  # unit: px
        self.val_hm_loss_min = 1e6
        self.val_total_loss_min = 1e6
        self.val_kps_loss_min = 1e6
        self.val_PCK_max = 0

        self.train_hm_loss = []
        self.train_kps_loss = []
        self.train_total_loss = []

    def process_training_output(self, hm_pred: torch.Tensor, hm_gn: torch.Tensor, kps_gn: torch.Tensor):
        """
        This function computes all loss functions and record current performance.

        params:
        @ hm_pred: 21 predicted heat maps of size stage x B x 21(kps) x 64(W) x 64(H)
        @ hm_gn: The ground truth heat maps of size B x 21(kps) x 64(W) x 64(H)
        @ kps_gn: The 2D key point annotations of size B x K x 2

        retval:
        @ hm_loss: heat map loss
        """

        # 1. Heat map loss: the Frobenius norm sqrt(sum((A-B).^2))
        # normalize heat maps to make each one sum to 1

        hm_loss = torch.tensor(0,dtype=torch.float).to(device)
        for stage in range(len(hm_pred)):
            for b in range(hm_pred[stage].size(0)):
                for c in range(hm_pred[stage].size(1)):
                    # hm_pred[stage][b][c] /= hm_pred[stage][b][c].sum()
                    hm_loss += self.hm_loss_fn(hm_pred[stage][b][c] / hm_pred[stage][b][c].sum(), hm_gn[b][c]) / config.batch_size  # size: a scalar

        # hm_loss = self.hm_loss_fn(hm_pred[-1], hm_gn.float()) / config.batch_size
        hm_pred_final = hm_pred[-1]
        # 2. 2D pose loss: the average distance between predicted key points and the ground truth
        #kps_pred = kornia.geometry.spatial_soft_argmax2d(hm_pred)  # size: (B*21) x 2
        kps_pred = dsnt.spatial_expectation2d(hm_pred_final, normalized_coordinates=False).view(-1,2) # size: B x 21 x 2

        kps_gn = kps_gn.view(-1, 2).to(device)  # change its size to (B*21) x 2
        kps_loss = self.pose2d_loss_fn(kps_pred, kps_gn) / hm_pred_final.size(0)  # loss function: Euclidean distance, size:
        total_loss = config.hm_loss_weight * hm_loss + config.kps_loss_weight * kps_loss

        self.num_train_iters += 1
        if self.num_train_iters % config.record_period == 0:
            print("{}/{} iterations have been finished".format(self.num_train_iters / config.record_period, config.num_iters))
            self.train_hm_loss.append(hm_loss.item())
            print("heat map loss ({:.2e}): {:.4f}".format(config.hm_loss_weight, hm_loss.item()))
            self.train_kps_loss.append(kps_loss.item())
            print("2D Pose Loss ({:.2e}): {:.4f}".format(config.kps_loss_weight, kps_loss.item()))
            self.train_total_loss.append(total_loss.item())
            print("Total Loss: {:.4f}\n".format(total_loss.item()))

            self.writer.add_scalar('HM_Loss/Training', hm_loss.item(), self.num_train_iters)
            self.writer.add_scalar('Pose2D_Loss/Training', kps_loss.item(), self.num_train_iters)
            self.writer.add_scalar('Total_Loss/Training', total_loss.item(), self.num_train_iters)

        return total_loss

    def eval(self):
        self.val_hm_loss.append(0)
        self.val_avg_PCK.append(0)
        self.val_kps_loss.append(0)
        self.val_total_loss.append(0)
        
        for i in range(21):
            self.val_PCK_of_each_joint[i].append(0)
            self.val_EPE_of_each_joint[i].append(0)

    def finish_eval(self):
        is_best_HM_loss, is_best_pose2d_loss, is_best_total_loss = False, False, False

        self.val_hm_loss[-1] /= config.num_train_samples
        self.val_total_loss[-1] /= config.num_train_samples
        
        for i in range(21):
            self.val_PCK_of_each_joint[i][-1] /= config.num_train_samples
            self.val_EPE_of_each_joint[i][-1] /= config.num_train_samples

        self.val_kps_loss[-1] /= config.num_train_samples
        self.val_avg_PCK[-1] /= (config.num_train_samples * 21)

        self.num_val_iters += 1
        self.writer.add_scalar('HM_Loss/Validation', self.val_hm_loss[-1], self.num_val_iters)
        self.writer.add_scalar('PCK/Validation', self.val_avg_PCK[-1], self.num_val_iters)
        self.writer.add_scalar('Pose2D_Loss/Validation', self.val_kps_loss[-1], self.num_val_iters)
        self.writer.add_scalar('Total_Loss/Validation', self.val_total_loss[-1], self.num_train_iters)
        
        if self.val_kps_loss_min > self.val_kps_loss[-1]:
            self.val_kps_loss_min = self.val_kps_loss[-1]
            is_best_pose2d_loss = True

        if self.val_hm_loss_min > self.val_hm_loss[-1]:
            self.val_hm_loss_min = self.val_hm_loss[-1]
            is_best_HM_loss = True

        if self.val_PCK_max < self.val_avg_PCK[-1]:
            self.val_PCK_max = self.val_avg_PCK[-1]

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
        hm_loss = torch.tensor(0, dtype=torch.float).to(device)
        for stage in range(len(hm_pred)):
            for b in range(hm_pred[stage].size(0)):
                for c in range(hm_pred[stage].size(1)):
                    # hm_pred[stage][b][c] /= hm_pred[stage][b][c].sum()
                    hm_loss += self.hm_loss_fn(hm_pred[stage][b][c] / hm_pred[stage][b][c].sum(),
                                               hm_gn[b][c]) / config.batch_size  # size: a scalar

        # hm_loss = self.hm_loss_fn(hm_pred[-1], hm_gn.float()) / config.batch_size
        hm_pred_final = hm_pred[-1]

        self.val_hm_loss[-1] += hm_loss.item()
        
        # size: B x K
        EPE_each_batch_each_joint, PCK_each_batch_each_joint = self.calc_PCK_EPE(hm_pred_final, kps_gn, 10)

        EPE_each_joint = torch.sum(EPE_each_batch_each_joint, dim=0)  # size: K
        PCK_each_joint = torch.sum(PCK_each_batch_each_joint, dim=0)  # size: K

        for i in range(21):
            self.val_PCK_of_each_joint[i][-1] += PCK_each_joint[i].item()
            self.val_EPE_of_each_joint[i][-1] += EPE_each_joint[i].item()

        kps_pred = dsnt.spatial_expectation2d(hm_pred_final, normalized_coordinates=False).view(-1,2)
        kps_gn = kps_gn.view(-1, 2).to(device)  # change its size to (B*21) x 2
        print(kps_pred.shape, kps_gn.shape)
        kps_loss = self.pose2d_loss_fn(kps_pred, kps_gn) / hm_pred_final.size(0)
        self.val_kps_loss[-1] += kps_loss.item()
        self.val_avg_PCK[-1] += torch.sum(PCK_each_joint).item()

    # probability of correct points
    def calc_PCK_EPE(self, hm_pred: torch.tensor, kps_gn: torch.tensor, th: int):
        """
        This function calculates PCK and EPE for each joint but returns average PCK and EPE over all joints.

        params:
        @ hm_pred: 21 predicted heat maps of size B x K(21 kps) x 64(W) x 64(H)
        @ kps_gn: The 2D key point annotations of size B x K x 2
        @ th: The threshold within which a predicted key point is seen as correct
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
        save_list(os.path.join(config.history_dir, config.model_name + '_val_avg_EPE.txt'), self.val_kps_loss)
        save_list(os.path.join(config.history_dir, config.model_name + '_val_avg_PCK.txt'), self.val_avg_PCK)

        save_multilist(os.path.join(config.history_dir, config.model_name + '_val_PCK_of_each_joint.txt'), self.val_PCK_of_each_joint)
        save_multilist(os.path.join(config.history_dir, config.model_name + '_val_EPE_of_each_joint.txt'),
                       self.val_EPE_of_each_joint)


if __name__=='__main__':
   pass
