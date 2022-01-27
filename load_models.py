import torch, torchvision

from torchvision import models
from torch import nn

import pruning_utils




class ImageNetNormalization(nn.Module):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(ImageNetNormalization, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return torchvision.transforms.functional.normalize(x, self.mean, self.std)
    


def load_resnet(MODEL_PATH, norm_layer=True, pruned=False):
    base_resnet = models.resnet18(pretrained=True)

    num_ftrs_in = base_resnet.fc.in_features
    num_ftrs_out = 10
    base_resnet.fc = nn.Linear(num_ftrs_in, num_ftrs_out)
    if norm_layer:
        resnet = torch.nn.Sequential(
            ImageNetNormalization(),
            base_resnet
        )
    else:
        resnet = base_resnet
        
    if pruned==True:
        modules_to_prune = pruning_utils.get_prunable_modules(resnet)
        pruning_utils.l1_prune(modules_to_prune, .0)
    print('yo')
    print(type(resnet))
    state_dict = torch.load(MODEL_PATH)
    resnet.load_state_dict(state_dict)
    model = resnet
    model.eval()
    return model

def load_vgg(MODEL_PATH, norm_layer=False, pruned=False):
    vgg16 = models.vgg16(pretrained=False)
    required_in_features = vgg16.classifier[-1].in_features

    num_classes = 10
    vgg16.classifier[-1] = nn.Linear(in_features=required_in_features, out_features=num_classes)
    if norm_layer:
        vgg16 = torch.nn.Sequential(
            ImageNetNormalization(),
            vgg16
        )
        
    if pruned==True:
        modules_to_prune = pruning_utils.get_prunable_modules(vgg16)
        pruning_utils.l1_prune(modules_to_prune, .0)
    state_dict = torch.load(MODEL_PATH)
    vgg16.load_state_dict(state_dict)
    vgg16.eval()
    return vgg16
    


def load_mobilenet(MODEL_PATH, norm_layer=True):

    mobilenet = models.mobilenet_v3_small(pretrained=True)

    num_ftrs_in = mobilenet.classifier[0].in_features
    num_ftrs_out = mobilenet.classifier[0].out_features
    mobilenet.classifier[0] = nn.Linear(num_ftrs_in, num_ftrs_out)

    num_ftrs_in = mobilenet.classifier[3].in_features
    num_ftrs_out = 10
    mobilenet.classifier[3] = nn.Linear(num_ftrs_in, num_ftrs_out)

    if norm_layer:
        mobilenet = torch.nn.Sequential(
            ImageNetNormalization(),
            mobilenet
        )
    else:
        mobilenet = torch.nn.Sequential(
            mobilenet
        )

    state_dict = torch.load(MODEL_PATH)
    mobilenet.load_state_dict(state_dict)

    model = mobilenet
    model.eval()
    
    return model