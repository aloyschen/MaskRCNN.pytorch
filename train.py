import os
import time
import config
import torch
from utils import encode, AverageTracker
from anchor import Anchor
from loss import MultiBoxLoss
import torch.optim as optim
from model.Retinanet import RetinaNet
from torch.utils.data import DataLoader
from transform import Augmentation
from COCODataset import COCODataset

# 指定使用GPU的Index
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index

def train():
    """
    Introduction
    ------------
        训练Retinanet模型
    """
    train_transform = Augmentation(size = config.image_size)
    train_dataset = COCODataset(config.coco_train_dir, config.coco_train_annaFile, config.coco_label_file, training = True, transform = train_transform)
    train_dataloader = DataLoader(train_dataset,batch_size = config.train_batch, shuffle = True, num_workers = 2, collate_fn = train_dataset.collate_fn)
    print("training on {} samples".format(train_dataset.__len__()))
    net = RetinaNet(config.num_classes, pre_train_path = config.resnet50_path)
    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr = config.learning_rate)
    criterion = MultiBoxLoss(alpha = config.focal_alpha, gamma = config.focal_gamma, num_classes = config.num_classes)
    anchors = Anchor(config.anchor_areas, config.aspect_ratio, config.scale_ratios)
    anchor_boxes = anchors(input_size = config.image_size)
    for epoch in range(config.Epochs):
        batch_time, loc_losses, conf_losses = AverageTracker(), AverageTracker(), AverageTracker()
        net.train()
        end = time.time()
        for idx, (image, gt_boxes, labels) in enumerate(train_dataloader):
            loc_targets, cls_targets = [], []
            image = image.cuda()
            loc_preds, cls_preds = net(image)
            batch_num = image.shape[0]
            for idx in range(batch_num):
                gt_box = gt_boxes[idx]
                label = labels[idx]
                loc_target, cls_target = encode(anchor_boxes, gt_box, label)
                loc_targets.append(loc_target)
                cls_targets.append(cls_target)
            loc_targets = torch.stack(loc_targets).cuda()
            cls_targets = torch.stack(cls_targets).cuda()
            loc_loss, cls_loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss = loc_loss + cls_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loc_losses.update(loc_loss.item(), image.size(0))
            conf_losses.update(cls_loss.item(), image.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if idx % config.print_freq == 0:
                print('Epoch: {}/{} Batch: {}/{} loc Loss: {:.4f} {:.4f} conf loss: {:.2f} {:.2f} Time: {:.4f} {:.4f}'.format(epoch, config.Epochs, idx, len(train_dataloader), loc_losses.val, loc_losses.avg, conf_losses.val, conf_losses.avg, batch_time.val, batch_time.avg))
        if epoch % config.save_freq == 0:
            print('save model')
            torch.save(net.state_dict(), config.model_dir + 'train_model_epoch{}.pth'.format(epoch + 1))


if __name__ == '__main__':
    train()