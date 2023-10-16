import os,sys
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import argparse
from eval.evaluator import Evaluator
import warnings
warnings.filterwarnings("error")
from torch.utils.data.dataloader import DataLoader
import torchsummary as summary
from model.yolov3 import DarkNet53
from dataloader.yololabeldata import *
from train.trainer import Trainer
from labeling.labeling import Labeling
from dataloader.data_transforms import *
from tensorboardX import SummaryWriter
import pynvml

import cv2
import glob
import onnx,onnxruntime

def get_memory_free_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024 ** 2

def get_memory_total_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.total // 1024 ** 2

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOV3-PYTORCH")
    parser.add_argument("--gpus", type=int, nargs='+', default=[], help="List of device ids.")
    parser.add_argument('--cfg', dest='cfg', help="model config path",
                        default=None, type=str)
    parser.add_argument('--checkpoint', dest='checkpoint', help = "the path of checkpoint",
                        default=None, type=str)
    parser.add_argument('--output', dest='output', help = "output file name",
                        default='sample', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def collate_fn(batch):
    batch = [data for data in batch if data is not None]
    #skip invalid frames
    if len(batch) == 0:
        return

    imgs, targets, anno_path = list(zip(*batch))

    imgs = torch.stack([img for img in imgs])
    
    if targets[0] is None or anno_path[0] is None:
        return imgs, None, None

    for i, boxes in enumerate(targets):
        boxes[:, 0] = i
    targets = torch.cat(targets,0)

    return imgs, targets, anno_path

def save_frames_from_video(video_path, output_path):
    video = cv2.VideoCapture(video_path)

    # frame save
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0
    pbar = tqdm(total=int(video.get(cv2.CAP_PROP_FRAME_COUNT)), desc='frame saved..', bar_format='{l_bar}{bar:20}{r_bar}')
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame_path = os.path.join(output_path, f"track_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

        frame_count += 1
        pbar.update(1)

    video.release()

    file_paths = glob.glob(f"{output_path}/*.jpg")
    file_paths = sorted(file_paths, key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    # txt file save
    with open(output_path+'/labeling.txt', 'w') as f:
        for file_path in file_paths:
            file_name = file_path.split("/")[-1].replace('.jpg','')
            f.write(f"{file_name}\n")

def autolabel(cfg_param = None, using_gpus = None):
    print("label")
    os.makedirs(f'./result/{args.output}', exist_ok=True)
    transforms = get_transformations(cfg_param, is_train = False)

    # mp4 -> jpg
    save_frames_from_video('./datasets/track.mp4', './datasets/labeling')

    data = Yolodata(mode = 'test', 
                    transform = transforms, 
                    cfg_param = cfg_param)
    demo_loader = DataLoader(data, batch_size = 1, 
                             num_workers = 0, 
                             pin_memory = True, 
                             drop_last = False, 
                             shuffle = False, 
                             collate_fn=collate_fn)
    
    model = DarkNet53(args.cfg, cfg_param)
    model.eval()
    
    if args.checkpoint is not None:
        print("load pretrained model", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])

    if len(using_gpus) == 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if device == torch.device('cuda'):
        print("device is cuda")
    elif device == torch.device('cpu'):
        print('device is cpu')
    
    model = model.to(device)
    model.eval()
    
    if args.checkpoint is not None:
        darknet_weights_name = args.checkpoint.replace(".pth", ".weights")
    model.save_darknet_weights(darknet_weights_name, cutoff=-1)

    torch.backends.cudnn.benchmark = True

    labeling = Labeling(model, data, demo_loader, device, cfg_param)
    
    labeling.run(args.output)


if __name__ == "__main__":
    args = parse_args()
    cfg_data = parse_hyperparam_config(args.cfg)
    cfg_param = get_hyperparam(cfg_data)
    
    # multi-gpu
    using_gpus = [int(g) for g in args.gpus]

    autolabel(cfg_param, using_gpus)
    print("finish")
