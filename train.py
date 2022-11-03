import json
import os

import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import references.detection.transforms as T
import sys
sys.path.append('./references/detection')
from references.detection.engine import train_one_epoch, evaluate


class DataSet:
    def __init__(self, image_dir, json_dir, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = json_dir
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(image_dir)))
        self.masks = list(sorted(os.listdir(json_dir)))

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        boxes = []
        json_data = json.load(open(os.path.join(self.mask_dir, self.masks[idx])))['shapes']
        for item in json_data:
            item = item['points']
            if item[0][0] > item[1][0]:
                item[0][0], item[1][0] = item[1][0], item[0][0]
            if item[0][1] > item[1][1]:
                item[0][1], item[1][1] = item[1][1], item[0][1]
            boxes.append(sum(item, []))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones(len(boxes), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros(len(boxes), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def train(train_iter, test_iter, net, optimizer, lr_scheduler, device, num_epochs):
    net = net.to(device)
    print('training on', device)

    # train for one epoch, printing every 10 iterations
    for i in range(num_epochs):
        train_one_epoch(net, optimizer, train_iter, device, num_epochs, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(net, test_iter, device=device)

    pass
