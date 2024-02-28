import json
import pathlib
import copy

import numpy as np
import torch
import torch.utils.data
from torchvision.ops import masks_to_boxes
from torchvision import transforms
from torchvision.tv_tensors import BoundingBoxes, Mask
from PIL import Image, ImageDraw

import augmentation
from cjm_torchvision_tfms.core import CustomRandomIoUCrop, ResizeMax, PadSquare


class DroneImages(torch.utils.data.Dataset):
    def __init__(self, root: str = 'data'):
        self.root = pathlib.Path(root)
        
        self.parse_json(self.root / 'old_descriptor.json')
        self.old_ids, self.old_images, self.old_polys, self.old_bboxes = self.ids, self.images, self.polys, self.bboxes
        self.parse_json(self.root / 'new_descriptor.json')
        self.new_ids, self.new_images, self.new_polys, self.new_bboxes = self.ids, self.images, self.polys, self.bboxes

    def parse_json(self, path: pathlib.Path):
        """
        Reads and indexes the descriptor.json

        The images and corresponding annotations are stored in COCO JSON format. This helper function reads out the images paths and segmentation masks.
        """
        with open(path, 'r') as handle:
            content = json.load(handle)

        self.ids = [entry['id'] for entry in content['images']]
        self.images = {entry['id']: self.root / pathlib.Path(entry['file_name']).name for entry in content['images']}

        # add all annotations into a list for each image
        self.polys = {}
        self.bboxes = {}
        for entry in content['annotations']:
            image_id = entry['image_id']
            self.polys.setdefault(image_id, []).append(entry['segmentation'])
            self.bboxes.setdefault(image_id, []).append(entry['bbox'])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a drone image and its corresponding segmentation mask.

        The drone image is a tensor with dimensions [H x W x C=5], where
            H - height of the image
            W - width of the image
            C - (R,G,B,T,H) - five channels being red, green and blue color channels, thermal and depth information

        The corresponding segmentation mask is binary with dimensions [H x W].
        """
        if index <= 322:
            image_id = self.old_ids[index]
        else:
            image_id = self.new_ids[index]

        # deserialize the image from disk
        if index <= 322:
            x = np.load(self.old_images[image_id])
        else:
            x = np.load(self.new_images[image_id])

        if index <= 322:
            polys = self.old_polys[image_id]
            bboxes = self.old_bboxes[image_id]
            masks = []
        else:
            polys = self.new_polys[image_id]
            bboxes = self.new_bboxes[image_id]
            masks = []
            
        # generate the segmentation mask on the fly
        for poly in polys:
            mask = Image.new('L', (x.shape[1], x.shape[0],), color=0)
            draw = ImageDraw.Draw(mask)
            if index <= 322:
                draw.polygon([i + ((ind + 1) % 2) * 60 for ind,i in zip(range(len(poly[0])), poly[0])], fill=1, outline=1)
            else:
                draw.polygon(poly[0], fill=1, outline=1)
                
            masks.append(np.array(mask))

        masks = torch.tensor(np.array(masks))
        labels = torch.tensor([1 for a in polys], dtype=torch.int64)

        boxes = torch.tensor(bboxes, dtype=torch.float)
        
        # bounding boxes are given as [x, y, w, h] but rcnn expects [x1, y1, x2, y2]
        boxes[:, 0] += 60 if index <= 322 else 0
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        x = torch.tensor(x, dtype=torch.float).permute((2, 0, 1))
        boxes = BoundingBoxes(data=masks_to_boxes(masks), 
                            format='xyxy', canvas_size= x.size()[-2:])
        ## data augmentation
        # x, masks = augmentation.transform(x, masks, 
        #                                   None, 
        #                                   crop_size= (1608, 2022), 
        #                                   scale_size=(0.7, 2.0), 
        #                                   augmentation=True)

        y = {
            'boxes': boxes,  # FloatTensor[N, 4]
            'labels': labels,  # Int64Tensor[N]
            'masks': masks,  # UIntTensor[N, H, W]
        }

        
        with_augmentation = False

        if with_augmentation:

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
            

            cropped_img, targets = iou_crop(x, y)
            resized_img, targets = resize_max(cropped_img, targets)
            padded_img, targets = pad_square(resized_img, targets)

            # Ensure the padded image is the target size
            resize = transforms.v2.Resize([train_sz_x, train_sz_y], antialias=True)
            resized_padded_img, targets = resize(padded_img, targets)
            #print(type(targets))
            #targets = resize(targets)
            sanitized_img, targets = transforms.v2.SanitizeBoundingBoxes()(resized_padded_img, targets)

            # Random color jitter
            if torch.rand(1) > 0.2:
                color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))  #For PyTorch 1.9/TorchVision 0.10 users
                # color_transform = transforms.ColorJitter.get_params((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
                sanitized_img[:3, :, :] = color_transform(sanitized_img[:3, :, :])

            # Random Gaussian filter
            if torch.rand(1) > 0.5:
                sanitized_img = transforms.GaussianBlur([3,5], (0.15, 1.15))(sanitized_img)

        #norm_mean = [130.0, 135.0, 135.0, 118.0, 118.0]
        #norm_std = [44.0, 40.0, 40.0, 30.0, 21.0]
        
        # DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1.transforms
        norm_mean = [0.485, 0.456, 0.406,  0.4627, 0.4627]
        norm_std = [0.229, 0.224, 0.225, 0.118, 0.08]
        x = x / 255.

        norm=transforms.Normalize(
            mean= norm_mean,
            std= norm_std)

        if with_augmentation:
            sanitized_img = sanitized_img / 255.
            normalized_img = norm(sanitized_img)
        else:
            normalized_img = norm(x)
            targets = y


        return normalized_img, targets
