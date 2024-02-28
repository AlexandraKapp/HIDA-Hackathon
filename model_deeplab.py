from torch import nn

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.segmentation import deeplabv3_resnet101,deeplabv3_resnet50,DeepLabV3_ResNet101_Weights, DeepLabV3_ResNet50_Weights

#import segmentation_models_pytorch as smp

def DeepLabV3(in_channels=5, num_classes=2, **kwargs):
    # if image_mean is None:
    #     image_mean = [0.485, 0.456, 0.406, 0.5, 0.5]
    # if image_std is None:
    #     image_std = [0.229, 0.224, 0.225, 0.225, 0.225]

    # model = smp.DeepLabV3(
    #     encoder_name="vgg16",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    #     in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #     classes=num_classes,                      # model output channels (number of classes in your dataset)
    # )

    #model = deeplabv3_resnet101(
    #    weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
    model = deeplabv3_resnet50(
        weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        )

    # change input layer
    model.backbone.conv1  = nn.Conv2d(in_channels, 64, 7, 2, padding=3, bias=False)


    # change last layer
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
    model.aux_classifier[-1] = nn.Conv2d(256, num_classes, 1)

    #print(model)
    return model
