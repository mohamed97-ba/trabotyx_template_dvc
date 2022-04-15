import numpy as np
import os
from os.path import join, split, isdir, isfile, abspath
import torch
from PIL import Image
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import skimage.io as io
import tqdm
import PIL
import torchvision.datasets as dset
from torch.autograd import Variable


def get_loader(root_dir, resize, batch_size,
               num_thread=4, pin=False, shuffle=True):
    
    transform = transforms.Compose([
            transforms.Resize((resize, resize)),#   Not used for current version.
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation((np.random.randint(low=0, high=30))),
            transforms.GaussianBlur([random.randrange(1, 8, 2)]),
        ])
    dataset = dset.ImageFolder(root=root_dir, transform=transform, )
    
   
    if shuffle is True:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_thread,
                                 pin_memory=pin)
    else:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_thread,
                                 pin_memory=pin)
    
    return data_loader


    