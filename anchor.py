import torch
import numpy as np
import torch.nn as nn
from itertools import product as product

class Anchor(nn.Module):
    def __init__(self, anchor_areas, aspect_ratio, scale_ratios):
        super(Anchor, self).__init__()
        # p3 - p7对应的anchor面积
        self.anchor_areas = anchor_areas
        self.aspect_ratio = aspect_ratio
        self.scale_ratios = scale_ratios


    def forward(self, input_size):
        """
        Introduction
        ------------
            生成每个feature map层的anchor boxes
        Parameters
        ----------
            input_size: 输入尺寸
        Returns
        -------
            boxes: 所有特征层的anchor boxes
        """
        anchor_boxes = []
        feature_map_num = len(self.anchor_areas)
        feature_map_sizes = [(np.ceil(input_size / pow(2, i + 3)), np.ceil(input_size / pow(2, i + 3))) for i in range(feature_map_num)]
        for index, feature_map_size in enumerate(feature_map_sizes):
            for i, j in product(range(int(feature_map_size[0])), repeat = 2):
                cx = (i + 0.5) * input_size / feature_map_size[0]
                cy = (j + 0.5) * input_size / feature_map_size[0]
                s = self.anchor_areas[index]
                for ar in self.aspect_ratio:
                    h = np.sqrt(s / ar)
                    w = ar * h
                    for sr in self.scale_ratios:
                        anchor_h = h * sr
                        anchor_w = w * sr
                        anchor_boxes.append([cx, cy, anchor_w, anchor_h])
        return torch.Tensor(anchor_boxes)