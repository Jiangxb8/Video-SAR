import torch
import train
from references.detection import utils
import matplotlib.pyplot as plt
import cv2 as cv
import numpy
import copy


def draw_boxes(image, boxes, color):
    for box in boxes:
        cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), color)


def show_results(dataset, model, batch_size=1, device=torch.device('cpu')):
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)
    model = model.to(device)
    model.eval()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        pred_results = model(images)
        for i in range(len(images)):
            image = images[i].cpu().numpy()
            image_real = copy.deepcopy(image)
            image_real = numpy.transpose(image_real, [1, 2, 0])
            image_pred = copy.deepcopy(image)
            image_pred = numpy.transpose(image_pred, [1, 2, 0])
            draw_boxes(image_real, targets[i]['boxes'].cpu().detach().numpy().astype(int), (0, 0, 1))
            draw_boxes(image_pred, pred_results[i]['boxes'].cpu().detach().numpy().astype(int), (1, 0, 0))
            cv.imshow('real', image_real)
            cv.imshow('pred', image_pred)
            cv.waitKey(10)
    cv.waitKey()


num_classes = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = train.get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load("./result/net.pt"))

dataset = train.DataSet('./data/images', './data/json', train.get_transform(train=False))
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

# data = iter(data_loader).__next__()
# model.eval()
# pred_boxes = model(data[0])[0]['boxes'].detach().numpy().astype(int)
#
# image = data[0][0].numpy()
# boxes = data[1][0]['boxes'].numpy().astype(int)
# image = numpy.transpose(image, [1, 2, 0])
# draw_boxes(image, boxes, (0, 0, 1))
#
# # draw_boxes(image, pred_boxes, (1, 0, 0))
#
# cv.imshow('image', image)
#
# cv.waitKey()

show_results(dataset, model, batch_size=1, device=device)
