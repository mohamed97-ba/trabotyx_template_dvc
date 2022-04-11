import argparse
import os
import random
import shutil
import time
from os.path import isfile, join, split
import math
import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torch.optim
import tqdm
import yaml
import cv2
from torch.optim import lr_scheduler
from logger import Logger

from dataloader import get_forward
from network import Net
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


parser = argparse.ArgumentParser(description='Stabber Testing')
# arguments from command line
parser.add_argument('--config', default="./config.yml", help="path to config file")
parser.add_argument('--resume', default='result/reproduce1.0/model_best.pth', help='path to config file')
parser.add_argument('--tmp', default="", help='tmp')
args = parser.parse_args()

assert os.path.isfile(args.config)
CONFIGS = yaml.safe_load(open(args.config))

# merge configs
if args.tmp != "" and args.tmp != CONFIGS["MISC"]["TMP"]:
    CONFIGS["MISC"]["TMP"] = args.tmp

CONFIGS["OPTIMIZER"]["WEIGHT_DECAY"] = float(CONFIGS["OPTIMIZER"]["WEIGHT_DECAY"])
CONFIGS["OPTIMIZER"]["LR"] = float(CONFIGS["OPTIMIZER"]["LR"])

os.makedirs(CONFIGS["MISC"]["TMP"], exist_ok=True)
logger = Logger(os.path.join(CONFIGS["MISC"]["TMP"], "log.txt"))

logger.info(CONFIGS)


def main():
    logger.info(args)
    assert os.path.isdir(CONFIGS["DATA"]["DIR"])

    model = Net()

    if CONFIGS["TRAIN"]["DATA_PARALLEL"]:
        logger.info("Model Data Parallel")
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])

    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint["state_dict"])
    print('Epoch:', checkpoint['epoch'])
    # print('Threshold:', checkpoint['threshold'])

    val_loader = get_forward(CONFIGS["DATA"]["FORWARD_DIR"], CONFIGS['MODEL']["RESIZE"],
                            CONFIGS['MODEL']["INFLATION"],
                            batch_size=1, num_thread=CONFIGS["DATA"]["WORKERS"], test=True, split='val', shuffle=False)

    logger.info("Data loading done.")

    logger.info("Start training.")

    validate(val_loader, model)


visualise = True
def validate(val_loader, model):
    # switch to evaluate mode
    model.eval()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    total_loss = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        bar = tqdm.tqdm(val_loader)
        iter_num = len(val_loader.dataset) // 1

        for i, data in enumerate(bar):

            images, dim = data
            if CONFIGS["TRAIN"]["DATA_PARALLEL"]:
                images = Variable(images).cuda()

            else:
                images = Variable(images).cuda(device=CONFIGS["TRAIN"]["GPU_ID"])

            pred = model(images)

            output = torch.sigmoid(pred).detach().cpu().numpy().squeeze(1)

            y_pred.extend(np.round(output))

            if visualise == True:

                if not os.path.exists(os.path.join("forward_pred")):
                    os.makedirs(os.path.join("forward_pred"))
                #
                img = torch.nn.functional.interpolate(images, (640, 640), mode='bilinear', align_corners=True)
                img = img[0].detach().cpu().numpy()
                img = np.transpose(img, (1, 2, 0))
                img = (img*std)+mean
                img = np.float32(img*255)
                img = img.astype(np.uint8)
                img = img.copy()
                if y_pred[-1] == 1:
                    cv2.putText(img, "Pred:Stab", (int(10), int(30)), cv2.FONT_HERSHEY_SIMPLEX,
                                1, color=(0, 255, 0), thickness=2)
                else:

                    cv2.putText(img, "Pred:Don't Stab", (int(10), int(30)), cv2.FONT_HERSHEY_SIMPLEX,
                                1, color=(0, 0, 255), thickness=2)

                cv2.imwrite("forward_pred/img"+str(i)+'.jpg', img.astype(np.uint8))

if __name__ == '__main__':
    main()
