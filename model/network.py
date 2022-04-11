import torch
import numpy as np 
import torch.nn as nn

from model.backbone.fpn import FPN101, FPN50, FPN18, ResNext50_FPN
from model.backbone.mobilenet import MobileNet_FPN
from model.backbone.vgg_fpn import VGG_FPN
from model.backbone.res2net import res2net50_FPN

# from model.dht import DHT_Layer

class Net(nn.Module):
    def __init__(self, backbone, num_classes):
        super(Net, self).__init__()
        self.num_classes = num_classes
        if backbone == 'resnet18':
            self.backbone = FPN18(pretrained=True, output_stride=32)
            output_stride = 32
        if backbone == 'resnet50':
            self.backbone = FPN50(self.num_classes, output_stride=16, pretrained=True )
            output_stride = 16
        if backbone == 'resnet101':
            self.backbone = FPN101(output_stride=16)
            output_stride = 16
        if backbone == 'resnext50':
            self.backbone = ResNext50_FPN(output_stride=16)
            output_stride = 16
        if backbone == 'vgg16':
            self.backbone = VGG_FPN()
            output_stride = 16
        if backbone == 'mobilenetv2':
            self.backbone = MobileNet_FPN()
            output_stride = 32
        if backbone == 'res2net50':
            self.backbone = res2net50_FPN()
            output_stride = 32
        
        if backbone == 'mobilenetv2':

            self.last_conv = nn.Sequential(
                nn.Conv2d(128, 1, 1)
            )
        else:

            self.last_conv = nn.Sequential(
                nn.Conv2d(256, 3, 1)
            )

        self.convb = nn.Conv2d(3, 1, kernel_size=1)

    def upsample_cat(self, p1, height, width):
        p1 = nn.functional.interpolate(p1, size=(height, width), mode='bilinear', align_corners=True)

        return p1

    def forward(self, x):
        b, c, h, w = x.shape
        p1 = self.backbone(x)

        # p1 = self.dht_detector1(p1)
        # p2 = self.dht_detector2(p2)
        # p3 = self.dht_detector3(p3)
        # p4 = self.dht_detector4(p4)

        p1 = self.last_conv(p1)
        logist = self.upsample_cat(p1, h, w)
        # print(cat.shape)
        # logist = self.last_conv(cat)
        # binary = self.convb(logist)
        # binary = self.upsample_cat(binary,h,w)

        return logist
