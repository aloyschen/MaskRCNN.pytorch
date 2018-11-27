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
    loc_xy = ((boxes[:, :2] + boxes[:, 2:]) / 2 - anchors[:, :2]) / anchors[:, 2:]
    loc_wh = torch.log((boxes[:, 2:] - boxes[:, :2]) / anchors[:, 2:])
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

def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors, 4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


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

def one_hot_embedding(labels, num_classes):
    """
    Embedding labels to one-hot form.
    Args:
    :param labels: (LongTensor) class label, sized [N,].
    :param num_classes: (int) number of classes.
    :return:
            (tensor) encoded labels, size [N, #classes].
    """
    y = torch.eye(num_classes)  # [D, D]
    return y[labels]  # [N, D]