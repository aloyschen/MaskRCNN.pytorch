import cv2
import torch
import config
import torch.nn as nn
import numpy as np
from anchor import Anchor
import matplotlib.pyplot as plt
from utils import decode, nms
from model.Retinanet import RetinaNet
import torchvision.transforms as transforms
def box_nms(bboxes, scores, threshold=0.5, mode='union'):
    '''Non maximum suppression.
    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) bbox scores, sized [N,].
      threshold: (float) overlap threshold.
      mode: (str) 'union' or 'min'.
    Returns:
      keep: (tensor) selected indices.
    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2-x1+1) * (y2-y1+1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1+1).clamp(min=0)
        h = (yy2-yy1+1).clamp(min=0)
        inter = w*h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)
class Detect(nn.Module):
    def __init__(self, num_classes, top_k, conf_thresh, nms_thresh):
        """
        Introduction
        ------------
            模型的预测模块
        Parameters
        ----------
            num_classes: 数据集类别数量
            top_k: 每种类别保留多少个box
            conf_thresh: 物体概率的阈值
            nms_thresh: nms阈值
        """
        super(Detect, self).__init__()
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.top_k = top_k


    def forward(self, predictions, anchors):
        """
        Introduction
        ------------
            分别对每种类别的Box做nms, 选择每种类别的保留topk box
        Parameters
        ----------
            loc_data: 预测的box坐标
            conf_data: 预测的物体概率值
            anchors: anchor box坐标
        Returns
        -------
            output: 每种类别的topk box坐标和类别
        """
        loc, conf = predictions
        conf = nn.Sigmoid()(conf)
        loc_data = loc.detach()
        conf_data = conf.detach()
        num = loc_data.shape[0]
        anchors = anchors.detach()
        num_anchors = anchors.shape[0]
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.reshape([num, num_anchors, self.num_classes]).transpose(2, 1)
        for index in range(num):
            decoded_boxes = decode(anchors, loc_data[index])
            conf_scores = conf_preds[index].clone()
            # 每个类别做nms
            for class_index in range(self.num_classes):
                class_mask = conf_scores[class_index].gt(self.conf_thresh)
                scores = conf_scores[class_index][class_mask]
                if scores.shape[0] == 0:
                    continue
                loc_mask = class_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[loc_mask].reshape([-1, 4])
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[index, class_index, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
        return output

def predict(image_file, model_path, draw_image = True):
    """
    Introduction
    ------------
        对图片进行预测
    Parameters
    ----------
        image_file: 图片路径
        model_path: model路径
    """
    image = cv2.imread(image_file)
    image_resized = cv2.resize(image, (config.image_size, config.image_size))
    image_tensor = transforms.ToTensor()(image_resized).unsqueeze(0).float()
    ckpt = torch.load(model_path, map_location = 'cpu')
    model = RetinaNet(config.num_classes)
    model.load_state_dict(ckpt)
    model.eval()
    anchors = Anchor(config.anchor_areas, config.aspect_ratio, config.scale_ratios)
    anchor_boxes = anchors(input_size = config.image_size)
    detector = Detect(config.num_classes, config.top_k, config.conf_thresh, config.nms_thresh)
    predictions = model(image_tensor)
    detections = detector(predictions, anchor_boxes)
    for j in range(detections.shape[1]):
        dets = detections[0, j, :]
        mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
        dets = torch.masked_select(dets, mask).view(-1, 5)

        if dets.shape[0] == 0:
            continue
        draw_bbox(image_resized, dets, j)
    image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    plt.imshow(image_resized)
    plt.show()


def draw_bbox(image, boxes, class_idx):
    """
    Introduction
    ------------
        绘制所有检测到的box
    parameters
    ----------
        image: 图片
        boxes: 检测到物体的坐标
        class_idx: 类别索引
    """
    for idx, box in enumerate(boxes):
        if box[0] > 0:
            print('score: {}'.format(box[0]))
            print('label: {}'.format(config.VOC_CLASSES[class_idx]))
            cv2.rectangle(image, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), config.colors[idx % 3], 2)
            cv2.putText(image, '{} {:.2f}'.format(config.VOC_CLASSES[class_idx], box[0]), (int(box[1] - 10), int(box[2]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return image

if __name__ == '__main__':
    predict('./dog.jpg', './train_model_epoch16.pth')