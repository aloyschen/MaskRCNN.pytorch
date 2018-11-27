gpu_index = "0"
top_k = 200
conf_thresh = 0.3
nms_thresh = 0.5
train_batch = 16
image_size = 600
num_classes = 20
focal_alpha = 0.25
focal_gamma = 2
Epochs = 200
print_freq = 10
learning_rate = 1e-3
save_freq = 3
aspect_ratio = [0.5, 1, 2]
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
scale_ratios = [1., pow(2, 1 / 3.), pow(2, 2 / 3.)]
anchor_areas = [32 * 32., 64 * 64., 128 * 128., 256 * 256., 512 * 512.]
voc_root = './data/VOCdevkit/'
resnet50_path = './data/resnet50-19c8e357.pth'
model_dir = './train_model/'
coco_label_file = './data/label_map.txt'
coco_train_dir = '/data0/dataset/coco/train2017'
coco_train_annaFile = '/data0/dataset/coco/annotations/instances_train2017.json'
VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']
coco_classes = ['person', 'bicycle', 'car', 'motorbike', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra','giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard','surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote','keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

