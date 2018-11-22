import torch

def box_iou(box1, box2):
    """
    Introduction
    ------------
        计算两个box的iou
    Parameters
    ----------
        box1: box的坐标[xmin, ymin, xmax, ymax], shape[box1_num, 4]
        box2: box的坐标[xmin, ymin, xmax, ymax], shape[box2_num, 4]
    Returns
    -------
        iou: 两个box的iou数值, shape[box1_num, box2_num]
    """
    box_num1 = box1.shape[0]
    box_num2 = box2.shape[0]
    max_xy = torch.min(box1[:, 2:].unsqueeze(1).expand(box_num1, box_num2, 2), box2[:, 2:].unsqueeze(0).expand(box_num1, box_num2, 2))
    min_xy = torch.max(box1[:, :2].unsqueeze(1).expand(box_num1, box_num2, 2), box2[:, :2].unsqueeze(0).expand(box_num1,box_num2, 2))
    inter_xy = torch.clamp((max_xy - min_xy), min = 0)
    inter_area = inter_xy[:, :, 0] * inter_xy[:, :, 1]
    box1_area = ((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])).unsqueeze(1).expand_as(inter_area)
    box2_area = ((box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])).unsqueeze(0).expand_as(inter_area)
    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou


def encode(anchors, boxes, labels):
    """
    Introduction
    ------------
        对anchor坐标和真实box坐标计算偏移量
        tx = (x - anchor_x) / anchor_w
        ty = (y - anchor_y) / anchor_h
        tw = log(w / anchor_w)
        th = log(h / anchor_h)
    Parameters
    ----------
        boxes: 所有真实box坐标
        labels: 所有真实box标签
        input_size: 输入图片的尺寸
    """
    # 将anchor的坐标转换为[xmin, ymin, xmax, ymax]
    anchors = torch.cat((anchors[:, :2] - anchors[:, 2:] / 2, anchors[:, :2] + anchors[:, 2:] / 2), 1)
    ious = box_iou(anchors, boxes)
    max_ious, max_ids = ious.max(1)
    # 每个anchor box对应的iou最大的ground truth box
    boxes = boxes[max_ids]
    loc_xy = (boxes[:, :2] - anchors[:, :2]) / anchors[:, 2:]
    loc_wh = torch.log(boxes[:, 2:] / anchors[:, 2:])
    loc_targets = torch.cat([loc_xy, loc_wh], 1)
    cls_targets = labels[max_ids] + 1
    cls_targets[max_ious < 0.5] = 0
    ignore = (max_ious > 0.4) & (max_ious < 0.5)
    cls_targets[ignore] = -1
    return loc_targets, cls_targets


def decode(anchors, loc_preds):
    """
    Introduction
    ------------
        对预测的结果进行坐标转换
    Parameters
    ----------
        loc_preds: 预测的box坐标值
        anchors: 每层特征产生的anchor box坐标
    """
    loc_xy = loc_preds[:, :2]
    loc_wh = loc_preds[:, 2:]
    xy = loc_xy * anchors[:, 2:] + anchors[:, :2]
    wh = torch.exp(loc_wh) * anchors[:, 2:]
    boxes = torch.cat([xy - wh / 2, xy + wh / 2], 1)
    return boxes

class AverageTracker:
    """
    Introduction
    ------------
        求平均值
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count