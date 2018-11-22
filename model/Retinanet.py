import torch
import torch.nn as nn
import torch.nn.init as init

from model.fpn import build_FPN_ResNet50

class RetinaNet(nn.Module):

    def __init__(self, num_classes, pre_train_path = None):
        """
        Introduction
        ------------
            提取多个尺度的融合特征，卷积回归预测box坐标和
        """
        super(RetinaNet, self).__init__()
        self.num_anchors = 9
        self.fpn = build_FPN_ResNet50()
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)
        self._init_weights()
        if pre_train_path is not None:
            ckpt = torch.load(pre_train_path)
            state_dict = self.fpn.state_dict()
            for key in ckpt.keys():
                if not key.startswith('fc'):
                    state_dict[key] = ckpt[key]


    def forward(self, input):
        features = self.fpn(input)
        loc_preds = []
        cls_preds = []
        for feature in features:
            loc_pred = self.loc_head(feature)
            cls_pred = self.cls_head(feature)
            loc_pred = loc_pred.permute(0, 2, 3, 1).reshape([input.shape[0], -1, 4])
            cls_pred = cls_pred.permute(0, 2, 3, 1).reshape([input.shape[0], -1, self.num_classes])
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)



    def _make_head(self, output_channels):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(256, output_channels, kernel_size = 3, stride = 1, padding = 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
