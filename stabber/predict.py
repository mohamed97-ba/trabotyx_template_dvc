import argparse
from distutils import extension
import os
import random
import shutil
import time
from os.path import isfile, join, split
import matplotlib.pyplot as plt
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
from torchvision import transforms
from torch.optim import lr_scheduler
# from logger import Logger

from dataloader import get_loader
from network import Net
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


parser = argparse.ArgumentParser(description='Stabber Testing')
# arguments from command line
parser.add_argument('--config', default="./config.yml", help="path to config file")
parser.add_argument('--resume', default='result/output_v1/model_v5.pt', help='path to config file')
parser.add_argument('--tmp', default="", help='tmp')
args = parser.parse_args()

assert os.path.isfile(args.config)
CONFIGS = yaml.safe_load(open(args.config))

# # merge configs
if args.tmp != "" and args.tmp != CONFIGS["MISC"]["TMP"]:
    CONFIGS["MISC"]["TMP"] = args.tmp



def main():
    assert os.path.isdir(CONFIGS["DATA"]["DIR"])

    model = Net()

    if CONFIGS["TRAIN"]["DATA_PARALLEL"]:
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])

    checkpoint = torch.load(args.resume)
    
    #model.load_state_dict(checkpoint["state_dict"])
    model.load_state_dict(checkpoint)

    #print('Epoch:', checkpoint['epoch'])
    # print('Threshold:', checkpoint['threshold'])
    #torch.save(model.state_dict(), 'result/output/model_v5.1.pt')
    #exit()

    val_loader = get_loader(CONFIGS["DATA"]["TEST_DIR"],CONFIGS['MODEL']["RESIZE"],
                            batch_size=1,
                            num_thread=CONFIGS["DATA"]["WORKERS"], shuffle=False)

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
    time_set = []
    with torch.no_grad():
        bar = tqdm.tqdm(val_loader)
        iter_num = len(val_loader.dataset) // 1
        id = 0
       
        for i, data in enumerate(bar):
            id += 1
            #images, image_ori, label, coords, name, new_coords = data
            images, label = data
            
            print(images.shape)
            
            images = Variable(images.type(torch.float32)).cuda(device=CONFIGS["TRAIN"]["GPU_ID"])
            label = Variable(label).cuda(device=CONFIGS["TRAIN"]["GPU_ID"])

            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            # pred = model(input)
            start_all = time.perf_counter()
            # start = time.time()
            pred = model(images)
            invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

            images = invTrans(images)
            output = torch.sigmoid(pred).detach().cpu().numpy().squeeze(1)
            # output = out.detach().cpu().numpy()

            torch.cuda.synchronize()
            end_all = time.perf_counter()
            # print(end_all-start_all)
            time_set.append(end_all-start_all)
            # print(time.time()-start)
            # exit()
            y_pred.extend(np.round(output))
            y_true.extend(label.detach().cpu().numpy())
            images = torch.squeeze(images)
           
            image = images.detach().cpu().numpy()
            img  = (image*255).astype(np.uint8)
           
            img = np.transpose(img, (2, 1, 0))
            if y_pred[-1] == 1 and y_true[-1] == 1:
                #img = cv2.imread('{}'.format(val_loader.dataset.imgs[i+1][0]), cv2.IMREAD_UNCHANGED)
                #image_name , extension= os.path.splitext('{}'.format(val_loader.dataset.imgs[i-1][0].split('/')[-1]))
                os.chdir('/home/med-ba/trabotyx_template_dvc/stabber/visualise_stab_bboxtn/TP/')
                cv2.imwrite('image_{}.jpg'.format(id), img)
            elif y_pred[-1] == 1 and y_true[-1] == 0:
               # img = cv2.imread('{}'.format(val_loader.dataset.imgs[i+1][0]), cv2.IMREAD_UNCHANGED)
               # image_name , extension= os.path.splitext('{}'.format(val_loader.dataset.imgs[i-1][0].split('/')[-1]))
                os.chdir('/home/med-ba/trabotyx_template_dvc/stabber/visualise_stab_bboxtn/FP/')
                cv2.imwrite('image_{}.jpg'.format(id), img)
            elif y_pred[-1] == 0 and y_true[-1] == 0:
                os.chdir('/home/med-ba/trabotyx_template_dvc/stabber/visualise_stab_bboxtn/TN/')
                cv2.imwrite('image_{}.jpg'.format(id), img)
            else:
                os.chdir('/home/med-ba/trabotyx_template_dvc/stabber/visualise_stab_bboxtn/FN/')
                cv2.imwrite('image_{}.jpg'.format(id), img)
            # print(images[0])
            # images = images[0].numpy()
            # images = images.astype(np.uint8)
            # if y_true[-1] == 1:
            
            #   cv2.imwrite('/home/med-ba/trabotyx_template_dvc/stabber/visualise_stab_bboxtn', images)
            if visualise == False:

                if not os.path.exists(os.path.join("visualise_stab_bboxtn")):
                    os.makedirs(os.path.join("visualise_stab_bboxtn"))
                #
                # img = torch.nn.functional.interpolate(images, (size[0], size[1]), mode='bilinear', align_corners=True)
                img = image_ori[0].numpy()
                img = img.astype(np.uint8)
                if y_true[-1] == 1:
                    gt = "Stab"
                    color_gt = (0, 255, 0)
                else:
                    gt = "Don't Stab"
                    color_gt = (0, 0, 255)

                if y_pred[-1] == 1:
                    cv2.rectangle(img, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])),
                                  color=(0, 255, 0), thickness=2)
                    cv2.putText(img, "Pred:Stab", (int(coords[0]), int(coords[1])-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, "GT:"+gt, (int(coords[0]), int(coords[1])-30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color=color_gt, thickness=2)
                else:
                    cv2.rectangle(img, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])),
                                  color=(0, 0, 255), thickness=2)
                    cv2.putText(img, "Pred:Don't Stab", (int(coords[0]), int(coords[1])-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color=(0, 0, 255), thickness=2)
                    cv2.putText(img, "GT:"+gt, (int(coords[0]), int(coords[1])-30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color=color_gt, thickness=2)


                cv2.rectangle(img, (int(new_coords[0]), int(new_coords[1])), (int(new_coords[2]), int(new_coords[3])),
                                  color=(0, 0, 0), thickness=2)
                cv2.imwrite("visualise_stab_bboxtn/"+str(name[0])+".jpg", img.astype(np.uint8))

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        tpr = int(tp)/(int(tp) + int(fn) + 1e-8)
        tnr = int(tn)/(int(tn) + int(fp) + 1e-8)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        print('Validation result: ==== Accuracy: %.5f' % accuracy)
        print('Validation result: ==== F1 score: %.5f' % f1)
        print('Validation result: ==== Precision: %.5f' % precision)
        print('Validation result: ==== Recall: %.5f' % recall)
        print('Validation result: ==== TP: %.5f' % tp)
        print('Validation result: ==== FP: %.5f' % fp)
        print('Validation result: ==== FN: %.5f' % fn)
        print('Validation result: ==== TN: %.5f' % tn)
        print('Validation result: ==== True Positive Rate(TP/TP+FN): %.5f' % tpr)
        print('Validation result: ==== True Negative Rate(TN/TN+FP): %.5f' % tnr)
        print('Validation result: ==== Time): %.5f' % (np.mean(time_set[5:])/4))
        print('Validation result: ==== FPS): %.5f' % (4/np.mean(time_set[5:])))

if __name__ == '__main__':
    main()
