import os
import cv2
import torch
import numpy as np
from pycocotools.coco import COCO
import torch.utils.data as data


def get_label_map(label_file):
    """
    Introduction
    ------------
        annotation中标签不是连续的，需要重新映射处理
    Parameters
    ----------
        label_file: 标签对应关系
    """
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])
    return label_map


class COCOAnnotationTransform(object):
    def __init__(self, label_map_path):
        """
        Introduction
        ------------
            对coco数据集每张图片的box坐标和label进行处理
        Parameters
        ----------
            label_map_path: label映射关系文件
        """
        self.label_map = get_label_map(label_map_path)

    def __call__(self, target):

        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = self.label_map[obj['category_id']]
                final_box = list(np.array(bbox))
                final_box.append(label_idx)
                res += [final_box]
            else:
                print("no bbox problem!")

        return np.array(res)

class COCODataset(data.Dataset):
    def __init__(self, root, annFile, label_map_file, training = True, transform = None):
        self.root = root
        self.training = training
        self.transform = transform
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgToAnns.keys())
        self.target_transform = COCOAnnotationTransform(label_map_file)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds = img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(self.root, path))
        bbox = self.target_transform(target)
        if self.transform is not None:
            img, bbox, labels = self.transform(img, bbox[:, :4], bbox[:, 4])
        else:
            bbox = bbox[:, :4]
            labels = bbox[:, 4]
        return img, bbox, labels

    def collate_fn(self, batch):
        imgs = [torch.Tensor(x[0]) for x in batch]
        boxes = [torch.Tensor(x[1]) for x in batch]
        labels = [torch.LongTensor(x[2]) for x in batch]
        return torch.stack(imgs), boxes, labels

    def __len__(self):
        return len(self.ids)

