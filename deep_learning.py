import os

import torch
import torchvision
from torch.utils.data import DataLoader
import train
from references.detection import utils

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# our dataset has two classes only - background and person
num_classes = 2
dataset = train.DataSet('./data/images', './data/json', train.get_transform(train=True))
dataset_test = train.DataSet('./data/images', './data/json', train.get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

# get the model using our helper function
model = train.get_model_instance_segmentation(num_classes)

# load params
model.load_state_dict(torch.load("./result/net.pt"))

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0005,
                            momentum=0.9, weight_decay=0.0005)
# optimizer = torch.optim.Adam(params, lr=0.01)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=30,
                                               gamma=0.1)

# let's train it for 10 epochs
num_epochs = 1
if __name__ == '__main__':
    train.train(data_loader, data_loader_test, model, optimizer, lr_scheduler, device, num_epochs)
    # save
    torch.save(model.state_dict(), './result/net.pt')

