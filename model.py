from torch import nn

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform

def MaskRCNN(in_channels=5):
    # if image_mean is None:
    #     image_mean = [0.485, 0.456, 0.406, 0.5, 0.5]
    # if image_std is None:
    #     image_std = [0.229, 0.224, 0.225, 0.225, 0.225]
    model = maskrcnn_resnet50_fpn_v2(
        weights_backbone='IMAGENET1K_V1')
        #weights="MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1") 


    #norm_mean = [130.0, 135.0, 135.0, 118.0, 118.0]
    #norm_std = [44.0, 40.0, 40.0, 30.0, 21.0]

    # DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1.transforms
    norm_mean = [0.485, 0.456, 0.406,  0.4627, 0.4627]
    norm_std = [0.229, 0.224, 0.225, 0.118, 0.08]

    # change normalization
    model.transform = GeneralizedRCNNTransform(800, 1333, norm_mean, norm_std)

    # change input layer
    output_channels = 64
    input_channels = 5
    model.backbone.body.conv1  = nn.Conv2d(input_channels, output_channels, 7, 2, padding=3, bias=False)

    # change last layer
    mask_predictor_in_channels = 256  # == mask_layers[-1]
    mask_dim_reduced = 256 
    mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, 2)
    model.roi_heads.mask_predictor = mask_predictor


    #layers = list(model.children())[:-1]  
    #model = nn.Sequential(*layers)
    
    #features.extend([nn.Linear(num_features, num_classes)])  # add layer with output size num_classes
    #model.classifier = nn.Sequential(*features)  # Replace the model classifier
    #model.avgpool = model.avgpool

    return model
