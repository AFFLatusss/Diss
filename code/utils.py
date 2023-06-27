import torchvision.models as models
import torch
import time
import matplotlib.pyplot as plt
from torchvision import transforms



def load_efficient_net(pretrained=False):
    model = models.efficientnet_b0(pretrained=pretrained)
    model.eval()

    return model


def load_mobile_net(pretrained=False):
    model = models.mobilenet_v2(pretrained=pretrained)
    model.eval()

    return model

def preprocess():
    transform =  transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),])
    return transform




    
