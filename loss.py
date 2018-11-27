import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import one_hot_embedding

class FocalLoss(nn.Module):
    def __init__(self, alpha = None, gamma = 2):
        """
        Introduction
        ------------
            构建focal loss结构
        Parameters
        ----------
            alpha: focal loss的系数
            gamma: focal loss的指数系数
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        class_onehot = one_hot_embedding(targets.data.cpu().long(), 1 + self.num_classes)  # [N, 81]
        class_onehot = torch.Tensor(class_onehot[:, 1:]).cuda()  # exclude background

        prob = self.sigmoid(inputs)
        pt = prob * class_onehot + (1 - prob) * (1 - class_onehot)
        w = self.alpha * class_onehot + (1 - self.alpha) * (1 - class_onehot)
        w = w * (1 - pt).pow(self.gamma)
        return F.binary_cross_entropy_with_logits(inputs, class_onehot, w, size_average = False)
        # N = inputs.size(0)
        # C = inputs.size(1)
        # probs = self.sigmoid(inputs)
        #
        # # 对label做onehot处理
        # class_mask = torch.zeros([N, C]).cuda()
        # ids = targets.reshape([-1, 1])
        # class_mask.scatter_(1, ids.data, 1.)
        # alpha = self.alpha * class_mask + (1 - self.alpha) * (1 - class_mask)
        # focal_weight = probs * (1 - class_mask) + (1 - probs) * class_mask
        # focal_weight = alpha * focal_weight ** self.gamma
        #
        # return F.binary_cross_entropy_with_logits(inputs, class_mask, focal_weight, reduction = 'sum')

class MultiBoxLoss(nn.Module):
    def __init__(self, alpha = 0.25, gamma = 2, num_classes = 80):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.focal_loss = FocalLoss(alpha = alpha, gamma = gamma)

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        """
        Introduction
        ------------
            计算所有尺度下生成的pos anchor的loss
        Parameters
        ----------
            loc_preds: 预测的box坐标, [batch_size, anchor_num, 4]
            loc_targets: 实际box坐标, [batch_size, anchor_num, 4]
            cls_preds: 预测的box类别, [batch_size, anchor_num, class_num]
            cls_targets: 实际的box类别, [batch_size, anchor_num]
        """
        pos = cls_targets > 0
        num_pos = pos.long().sum()
        mask = pos.unsqueeze(2).expand_as(loc_preds)
        masked_loc_preds = loc_preds[mask].reshape([-1, 4])
        masked_loc_targets = loc_targets[mask].reshape([-1, 4])
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, reduction = 'sum')
        # 去除anchor和ground truth最大iou在[0.4, 0.5]区间内的anchor
        pos_neg = cls_targets > -1
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].reshape([-1, self.num_classes])
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])
        loc_loss = loc_loss / num_pos.float()
        cls_loss = cls_loss / num_pos.float()
        return loc_loss, cls_loss