#!/usr/bin/env python

import argparse
import random

import numpy as np
import torch
from torch import nn
import torch.optim
import torch.utils.data

from dataset import DroneImages
from model import MaskRCNN
from model_deeplab import DeepLabV3
from tqdm import tqdm
from torchmetrics import JaccardIndex
from cjm_torchvision_tfms.core import CustomRandomIoUCrop, ResizeMax, PadSquare
from torchvision import transforms
from torchvision.tv_tensors import BoundingBoxes, Mask


def collate_fn(batch) -> tuple:
    return tuple(zip(*batch))


            
def augment(x, y, device):
    # crop
    iou_crop = CustomRandomIoUCrop(min_scale=0.3, 
                        max_scale=1.0, 
                        min_aspect_ratio=1.0, 
                        max_aspect_ratio=1.0, 
                        sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                        trials=40, 
                        jitter_factor=0.25)

    
    # Set training image size
    train_sz_x = 2680
    train_sz_y = 3370
    # Create a `ResizeMax` object
    resize_max = ResizeMax(max_sz=train_sz_y)

    # Create a `PadSquare` object
    pad_square = PadSquare(shift=True, fill=0)
    
    augmented_images = []
    augmented_targets = []
    for i in range(0, len(x)):
        cropped_img, targets = iou_crop(x[i], y[i])
        resized_img, targets = resize_max(cropped_img, targets)
        padded_img, targets = pad_square(resized_img, targets)

        # Ensure the padded image is the target size
        resize = transforms.v2.Resize([train_sz_x, train_sz_y], antialias=True)
        resized_padded_img, targets = resize(padded_img, targets)
        
        #targets = resize(targets)
        sanitized_img, targets = transforms.v2.SanitizeBoundingBoxes()(resized_padded_img, targets)
        augmented_images.append(sanitized_img)
        augmented_targets.append(targets)
    # # Random color jitter
    # if torch.rand(1) > 0.2:
    #     color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))  #For PyTorch 1.9/TorchVision 0.10 users
    #     # color_transform = transforms.ColorJitter.get_params((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
    #     sanitized_img[:3, :, :] = color_transform(sanitized_img[:3, :, :])

    # # Random Gaussian filter
    # if torch.rand(1) > 0.5:
    #     sanitized_img = transforms.GaussianBlur([3,5], (0.15, 1.15))(sanitized_img)
    return augmented_images, augmented_targets


def instance_to_semantic_mask(pred, target):
    pred_mask = torch.stack([p['masks'].sum(dim=0).clamp(0., 1.).squeeze() for p in pred])  # [batch_size, width, height]
    target_mask = torch.stack([t['masks'].sum(dim=0).clamp(0., 1.).squeeze() for t in target])  # [batch_size, width, height]

    return pred_mask, target_mask

def instance_to_semantic_mask_deeplab(pred, target): 
    pred_mask = pred.argmax(dim=1)
    return pred_mask, target

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(hyperparameters: argparse.Namespace):
    deeplab = False

    # set fixed seeds for reproducible execution
    random.seed(hyperparameters.seed)
    np.random.seed(hyperparameters.seed)
    torch.manual_seed(hyperparameters.seed)

    # determines the execution device, i.e. CPU or GPU
    device = get_device()
    print(f'Training on {device}')

    # set up the dataset
    drone_images = DroneImages(hyperparameters.root)

    train_data, test_data = torch.utils.data.random_split(drone_images, [0.8, 0.2])

    # initialize MaskRCNN model
    if deeplab:
        model = DeepLabV3()
    else:
        model=MaskRCNN()
        #model = MaskRCNN(image_mean= [130.0, 135.0, 135.0, 118.0, 118.0], 
        #image_std= [44.0, 40.0, 40.0, 30.0, 21.0])
    model.to(device)

    # set up optimization procedure
    #optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters.lr)
    best_iou = 0.

    # start the actual training procedure
    for epoch in range(hyperparameters.epochs):
        # set the model into training mode
        model.train()
        train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=hyperparameters.batch,
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn)

        # training procedure
        train_loss = 0.0
        train_metric = JaccardIndex(task='binary')
        train_metric = train_metric.to(device)

        for i, batch in enumerate(tqdm(train_loader, desc='train')):
            x, label = batch


            if deeplab: #2900x3000
                pass
            #     x=torch.stack(x, dim=0).to(device)[:,:,:256,:1200]
            #     label = [{k: v.to(device) for k, v in l.items()} for l in label]
            #     label = torch.stack([p['masks'].sum(dim=0).squeeze() for p in label]).long()
                
            #     label = label [:, :256, :1200]   # [batch_size, width, height]
            #     label = label.to(device)
            #     model.zero_grad()
            
            #     outputs = model(x)
            #     loss = nn.CrossEntropyLoss()(outputs['out'], label)

            else:
                x, label = augment(x, label, device)
                x = list(image.to(device) for image in x)
                label = [{k: v.to(device) for k, v in l.items()} for l in label]
                

                model.zero_grad()
                losses = model(x, label)
                loss = sum(l for l in losses.values())

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # compute metric
            with torch.no_grad():
                model.eval()
                train_predictions = model(x)

                if deeplab:
                    train_metric(*instance_to_semantic_mask_deeplab(train_predictions['out'], label))
                else:
                    train_metric(*instance_to_semantic_mask(train_predictions, label))
                model.train()

        train_loss /= len(train_loader)

        # set the model in evaluation mode
        model.eval()
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=hyperparameters.batch, collate_fn=collate_fn)

        # test procedure
        test_metric = JaccardIndex(task='binary')
        test_metric = test_metric.to(device)

        for i, batch in enumerate(tqdm(test_loader, desc='test ')):

            x_test, test_label = batch
            x_test = list(image.to(device) for image in x_test)
            test_label = [{k: v.to(device) for k, v in l.items()} for l in test_label]

            with torch.no_grad():
                test_predictions = model(x_test)
                test_metric(*instance_to_semantic_mask(test_predictions, test_label))

        # output the losses
        print(f'Epoch {epoch}')
        print(f'\tTrain loss: {train_loss}')
        print(f'\tTrain IoU:  {train_metric.compute()}')
        print(f'\tTest IoU:   {test_metric.compute()}')

        # save the best performing model on disk
        if test_metric.compute() > best_iou:
            best_iou = test_metric.compute()
            print('\tSaving better model\n')
            torch.save(model.state_dict(), 'checkpoint.pt')
        else:
            print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', default=2, help='batch size', type=int)
    parser.add_argument('-e', '--epochs', default=50, help='number of training epochs', type=int)
    parser.add_argument('-l', '--lr', default=1e-4, help='learning rate of the optimizer', type=float)
    parser.add_argument('-s', '--seed', default=42, help='constant random seed for reproduction', type=int)
    parser.add_argument('--root', default='/hkfs/work/workspace_haic/scratch/qx6387-hida-hackathon-data/train', help='path to the data root', type=str)

    arguments = parser.parse_args()
    train(arguments)
