import argparse
import os
import random
import shutil
import time
from os.path import isfile, join, split

import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torch.optim
import tqdm
import yaml
from torch.optim import lr_scheduler
from logger import Logger

from torch.autograd import Variable
from dataloader import get_loader
from network import Net
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

parser = argparse.ArgumentParser(description='Stabber Training')
# arguments from command line

parser.add_argument('--config', default="./config.yml", help="path to config file")
parser.add_argument('--resume', default='', help='path to config file')
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

    if CONFIGS['TRAIN']['SEED'] is not None:
        random.seed(CONFIGS['TRAIN']['SEED'])
        torch.manual_seed(CONFIGS['TRAIN']['SEED'])
        cudnn.deterministic = True

    model = Net()

    if CONFIGS["TRAIN"]["DATA_PARALLEL"]:
        logger.info("Model Data Parallel")
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIGS["OPTIMIZER"]["LR"],
        weight_decay=CONFIGS["OPTIMIZER"]["WEIGHT_DECAY"]
    )

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=CONFIGS["OPTIMIZER"]["LR"],
    #     weight_decay=CONFIGS["OPTIMIZER"]["WEIGHT_DECAY"]
    # )

    # learning rate scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=CONFIGS["OPTIMIZER"]["STEPS"],
                                         gamma=CONFIGS["OPTIMIZER"]["GAMMA"])
    best_acc1 = 0
    if args.resume:
        if isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))


    # dataloader
    train_loader = get_loader(CONFIGS["DATA"]["DIR"], CONFIGS["DATA"]["LABEL_FILE"], CONFIGS['MODEL']["RESIZE"],
                              CONFIGS['MODEL']["INFLATION"], batch_size=CONFIGS["DATA"]["BATCH_SIZE"],
                              num_thread=CONFIGS["DATA"]["WORKERS"],
                              split='train')
    val_loader = get_loader(CONFIGS["DATA"]["VAL_DIR"], CONFIGS["DATA"]["VAL_LABEL_FILE"],CONFIGS['MODEL']["RESIZE"],
                              CONFIGS['MODEL']["INFLATION"], batch_size=CONFIGS["DATA"]["BATCH_SIZE"],
                            num_thread=CONFIGS["DATA"]["WORKERS"], split='val')

    logger.info("Data loading done.")

    # Tensorboard summary

    writer = SummaryWriter(log_dir=os.path.join(CONFIGS["MISC"]["TMPT"]))

    start_epoch = 0
    best_acc = best_acc1
    start_time = time.time()

    if CONFIGS["TRAIN"]["RESUME"] is not None:
        raise (NotImplementedError)

    if CONFIGS["TRAIN"]["TEST"]:
        validate(val_loader, model, 0, writer, args)
        return

    logger.info("Start training.")

    for epoch in range(start_epoch, CONFIGS["TRAIN"]["EPOCHS"]):
        # print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        train_acc = train(train_loader, model, optimizer, epoch + 1, writer, args)
        acc = validate(val_loader, model, epoch + 1, writer, train_acc, args)
        train_loader.dataset.data = train_loader.dataset.resample(epoch=epoch, rand=True)
        # print(train_loader.dataset.data[0])
        # return
        scheduler.step()

        if best_acc < acc:
            is_best = True
            best_acc = acc
            print('saved_acc:',epoch + 1)
        else:
            is_best = False

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, path=CONFIGS["MISC"]["TMP"])

        t = time.time() - start_time
        elapsed = DayHourMinute(t)
        t /= (epoch + 1) - start_epoch  # seconds per epoch
        t = (CONFIGS["TRAIN"]["EPOCHS"] - epoch - 1) * t
        remaining = DayHourMinute(t)

        logger.info("Epoch {0}/{1} finishied, auxiliaries saved to {2} .\t"
                    "Elapsed {elapsed.days:d} days {elapsed.hours:d} hours {elapsed.minutes:d} minutes.\t"
                    "Remaining {remaining.days:d} days {remaining.hours:d} hours {remaining.minutes:d} minutes.\t""lr:{3}".format(
            epoch, CONFIGS["TRAIN"]["EPOCHS"], CONFIGS["MISC"]["TMP"], optimizer.param_groups[0]['lr'], elapsed=elapsed,
            remaining=remaining))

    logger.info("Optimization done, ALL results saved to %s." % CONFIGS["MISC"]["TMP"])


# loss_fn = torch.nn.BCEWithLogitsLoss()
# class_weights = np.array([5])
# 0.01, 10, 25
# class_weights = torch.from_numpy(class_weights)
# class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()
loss_fn = torch.nn.BCEWithLogitsLoss()
# loss_fn = torch.nn.BCEWithLogitsLoss(weight=class_weights)

lambda_l2 = CONFIGS["OPTIMIZER"]["LAMBDA"]

def train(train_loader, model, optimizer, epoch, writer, args):
    # switch to train mode
    y_pred = []
    y_true = []
    model.train()
    bar = tqdm.tqdm(train_loader)
    iter_num = len(train_loader.dataset) // CONFIGS["DATA"]["BATCH_SIZE"]

    total_loss = 0
    for i, data in enumerate(bar):

        images, label = data
        if CONFIGS["TRAIN"]["DATA_PARALLEL"]:
            images = Variable(images).cuda()
            label = Variable(label).cuda()

        else:
            images = Variable(images).cuda(device=CONFIGS["TRAIN"]["GPU_ID"])
            label = Variable(label).cuda(device=CONFIGS["TRAIN"]["GPU_ID"])

        pred = model(images)

        output = torch.sigmoid(pred).detach().cpu().numpy().squeeze(1)

        l2 = torch.cat([x.view(-1) for x in model.net.parameters()])

        loss = loss_fn(pred.squeeze(1), label)

        y_pred.extend(np.round(output))
        y_true.extend(label.detach().cpu().numpy())

        if not torch.isnan(loss):

            total_loss += loss.item()

        else:
            logger.info("Warnning: loss is Nan.")

        loss += lambda_l2 * torch.norm(l2, p=2)

        # record loss
        bar.set_description('Training Loss:{}'.format(loss.item()))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    total_loss = total_loss / iter_num

    writer.add_scalar('train/total_loss', total_loss, epoch)
    writer.add_scalar('train/accuracy', accuracy, epoch)
    writer.add_scalar('train/precision', precision, epoch)
    writer.add_scalar('train/recall', recall, epoch)

    logger.info('Training result: ==== Accuracy: %.5f' % accuracy)

    return accuracy

def validate(val_loader, model, epoch, writer,train_acc, args):
    # switch to evaluate mode
    model.eval()
    total_loss = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        bar = tqdm.tqdm(val_loader)
        iter_num = len(val_loader.dataset) // 1

        for i, data in enumerate(bar):

            images, label = data

            if CONFIGS["TRAIN"]["DATA_PARALLEL"]:
                images = Variable(images).cuda()
                label = Variable(label).cuda()

            else:
                images = Variable(images).cuda(device=CONFIGS["TRAIN"]["GPU_ID"])
                label = Variable(label).cuda(device=CONFIGS["TRAIN"]["GPU_ID"])

            pred = model(images)

            output = torch.sigmoid(pred).detach().cpu().numpy().squeeze(1)
            l2 = torch.cat([x.view(-1) for x in model.net.parameters()])
            loss = loss_fn(pred.squeeze(1), label)



            if not torch.isnan(loss):
                total_loss += loss.item()
            else:
                logger.info("Warnning: loss is Nan.")

            loss += lambda_l2 * torch.norm(l2, p=2)
            writer.add_scalar('val/loss', loss.item(), epoch * iter_num + i)

            y_pred.extend(np.round(output))
            y_true.extend(label.detach().cpu().numpy())

        total_loss = total_loss / iter_num
        # bar.set_description('Val Loss:{}'.format(total_loss))
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        tpr = tp/(tp + fn + 1e-8)
        tnr = tn/(tn + fp + 1e-8)
        score = 0
        if tnr > 0.5 < tpr:
            if accuracy<train_acc:
                score = accuracy

        writer.add_scalar('val/total_loss', total_loss, epoch)
        writer.add_scalar('val/accuracy', accuracy, epoch)
        writer.add_scalar('val/precision', precision, epoch)
        writer.add_scalar('val/recall', recall, epoch)

        logger.info('Validation result: ==== Accuracy: %.5f' % accuracy)
        logger.info('Validation result: ==== F1 score: %.5f' % f1)
        logger.info('Validation result: ==== TP: %.5f' % tp)
        logger.info('Validation result: ==== FP: %.5f' % fp)
        logger.info('Validation result: ==== FN: %.5f' % fn)
        logger.info('Validation result: ==== TN: %.5f' % tn)

    return score


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(path, filename))
    if is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'model_best.pth'))


class DayHourMinute(object):

    def __init__(self, seconds):
        self.days = int(seconds // 86400)
        self.hours = int((seconds - (self.days * 86400)) // 3600)
        self.minutes = int((seconds - self.days * 86400 - self.hours * 3600) // 60)


if __name__ == '__main__':
    main()
    # print('trying vim')
