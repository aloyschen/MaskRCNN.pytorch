import os
import cv2
import torch
import config
import numpy as np
import xml.etree.ElementTree as ET
import torch.utils.data as data


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (list with tuple-string): imageset to use (eg. [('2007', 'train')])
        transform (callable, optional): transformation to perform on the input image
        target_transform (callable, optional): transformation to perform on the target `annotation`
            (eg: take in caption string, return tensor of word indices)
    """

    def __init__(self, root, image_set, transform=None, target_transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_set:
            rootpath = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, item):
        img, gt, h, w = self.pull_item(item)
        return img, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channel = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            target = np.c_[boxes, np.expand_dims(labels, axis=1)]

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id)

    def pull_anno(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)  # back original size
        return img_id[1], gt


    def collate_fn(self, batch):
        targets = []
        imgs = []
        for sample in batch:
            imgs.append(sample[0])
            targets.append(torch.FloatTensor(sample[1]))
        return torch.stack(imgs, 0), targets

class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(zip(config.VOC_CLASSES, range(len(config.VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                cur_pt = cur_pt if i % 2 == 0 else cur_pt
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # each elem: [xmin, ymin, xmax, ymax, label_ind]
        return res


class BaseTransform(object):
    def __init__(self, size=600):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        h, w, _ = image.shape
        image = cv2.resize(image, (self.size, self.size))
        boxes[:, 0] = boxes[:, 0] * self.size / w
        boxes[:, 1] = boxes[:, 1] * self.size / h
        boxes[:, 2] = boxes[:, 2] * self.size / w
        boxes[:, 3] = boxes[:, 3] * self.size / h
        image = image / 255
        return image, boxes, labels




def build_vocDataset(voc_root):
    image_set = [('2012', 'trainval')]
    target_transform = AnnotationTransform()
    dataset = VOCDetection(voc_root, image_set, transform = BaseTransform(), target_transform = target_transform)
    return dataset