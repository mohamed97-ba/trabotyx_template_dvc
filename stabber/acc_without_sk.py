import numpy as np
import argparse
import torch
from torch.autograd import Variable
import pycuda.autoinit
import tensorrt as trt
import pycuda.driver as cuda
import subprocess
import yaml
from dataloader import get_loader
from network import Net
import time
import os
from logger import Logger
import math
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def convert_torch2onnx(torch_model, images, onnx_file_path):

    torch.onnx.export(torch_model, images, onnx_file_path, verbose=False)


def convert_onnx2tensorrt(onnx_file_path, engine_file_path, d_type, batch):
    if d_type == np.float16:
        command = 'trtexec --onnx=' + onnx_file_path + ' --saveEngine=' + \
                  engine_file_path + ' --workspace=2048 --explicitBatch --fp16 --batch=' + str(batch)
    else:
        command = 'trtexec --onnx=' + onnx_file_path + ' --saveEngine=' + \
                  engine_file_path + ' --workspace=2048 --explicitBatch'

    subprocess.run(command, shell=True)


def sigmoid(x):
    return np.round(1 / (1 + np.exp(-x)))


def main(args):
    onnx_path = args.onnx_path
    trt_path = args.trt_path
    d_type = args.d_type
    regen = args.regen
    batch_size = args.batch

    input_batch = np.zeros((batch_size, 3, 300, 300), d_type)
    output = np.empty([batch_size, 1], dtype=d_type)

    model = Net()

    CONFIGS = yaml.safe_load(open(args.config))
    model = model.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])

    # if not os.path.exists(os.path.join(CONFIGS["MISC"]["TMP"], "acc_without_sk.txt")):
    #     open(os.path.join("acc_without_sk.txt"),"w+")

    logger = Logger("acc_without_sk.txt")
    logger.info(CONFIGS)

    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)
    # print('Epoch:', checkpoint['epoch'])

    if d_type == np.float16:
        model = model.half()

    if not os.path.isfile(onnx_path) or regen:
        print('enter onnx')
        exit()
        convert_torch2onnx(model, Variable(torch.from_numpy(input_batch)).cuda(), onnx_path)

    if not os.path.isfile(trt_path) or regen:
        convert_onnx2tensorrt(onnx_file_path = onnx_path, engine_file_path = trt_path, d_type = d_type, batch = batch_size)

    f = open(trt_path, "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # allocate device memory

    d_input = cuda.mem_alloc(1 * np.array(input_batch).nbytes)
    d_output = cuda.mem_alloc(1 * output.nbytes)
    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    def predict(batch):  # result gets copied into output
        # transfer input data to device
        cuda.memcpy_htod_async(d_input, batch, stream)
        # execute model
        context.execute_async_v2(bindings, stream.handle, None)
        # transfer predictions back
        cuda.memcpy_dtoh_async(output, d_output, stream)
        # syncronize threads
        stream.synchronize()

        return output

    val_loader = get_loader(CONFIGS["DATA"]["TEST_DIR"], CONFIGS["DATA"]["TEST_LABEL_FILE"],CONFIGS['MODEL']["RESIZE"],
                            CONFIGS['MODEL']["INFLATION"], batch_size=batch_size, num_thread=CONFIGS["DATA"]["WORKERS"],
                            test=True, split='val', shuffle=False)

    y_pred = []
    y_true = []
    time_set = []
    for data in val_loader:

        # images, image_ori, label, coords, name = data
        images, label = data
        images = np.asarray(images).astype(d_type)
        print(images.shape)
        # exit()
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_all = time.perf_counter()
        pred = predict(images)

        output_det = np.asarray(sigmoid(pred))

        torch.cuda.synchronize()
        end_all = time.perf_counter()

        print(end_all - start_all)

        time_set.append(end_all-start_all)
        y_pred.extend([output_det])
        y_true.extend(label)

    y_pred = np.asarray(y_pred).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)


    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = int(tp) / (int(tp) + int(fn) + 1e-8)
    tnr = int(tn) / (int(tn) + int(fp) + 1e-8)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    logger.info('Validation result: ==== Accuracy: %.5f' % accuracy)
    logger.info('Validation result: ==== F1 score: %.5f' % f1)
    logger.info('Validation result: ==== Precision: %.5f' % precision)
    logger.info('Validation result: ==== Recall: %.5f' % recall)
    logger.info('Validation result: ==== TP: %.5f' % tp)
    logger.info('Validation result: ==== FP: %.5f' % fp)
    logger.info('Validation result: ==== FN: %.5f' % fn)
    logger.info('Validation result: ==== TN: %.5f' % tn)
    logger.info('Validation result: ==== True Positive Rate(TP/TP+FN): %.5f' % tpr)
    logger.info('Validation result: ==== True Negative Rate(TN/TN+FP): %.5f' % tnr)
    logger.info('Validation result: ==== Time): %.5f' % (np.mean(time_set[5:])*(1/batch_size)))
    logger.info('Validation result: ==== FPS): %.5f' % (batch_size/(np.mean(time_set[5:]))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stabber Accelearation Testing')
    # arguments from command line
    parser.add_argument('--config', default="./config.yml", help="path to config file")
    parser.add_argument('--resume', default='result/resnet50_best/model_best.pt', help='path to config file')
    parser.add_argument('--onnx_path', default="onnx_model.onnx", help='onnx model path')
    parser.add_argument('--trt_path', default="testrtr.trt", help='tensorrt engine path')
    parser.add_argument('--d_type', default=np.float16, help='floating point operation')
    parser.add_argument('--batch', default=4, type=int, help='batch size for inference')
    parser.add_argument('--regen', default=False, help='regenerate both the models')

    args = parser.parse_args()
    main(args)

