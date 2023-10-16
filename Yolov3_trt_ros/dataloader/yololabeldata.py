import torch
from torch.utils.data import Dataset
import os,sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from util.tools import *

class Yolodata(Dataset):
    file_dir = ""
    anno_dir = ""
    file_txt = ""
    labeling_dir = "./datasets/labeling/"
    labeling_txt = "labeling.txt"
    #class_str = ['Car','Van','Truck','Pedestrian','Person','Cyclist','Tram','Miscellaneous']
    class_str = ['left','right','stop','crosswalk','uturn','traffic_light','xycar','ignore']
    num_class = None
    img_data = []
    def __init__(self, mode='test', transform=None, cfg_param=None):
        super(Yolodata, self).__init__()
        self.mode = mode
        self.transform = transform
        self.num_class = cfg_param['class']

        self.file_dir = self.labeling_dir
        self.file_txt = self.labeling_dir+self.labeling_txt
        
        img_names = []
        img_data = []
        with open(self.file_txt, 'r', encoding='UTF-8', errors='ignore') as f:
            img_names = [ i.replace("\n", "").split('/')[-1].replace('.txt','') for i in f.readlines()]

        for i in img_names:
            if os.path.exists(self.file_dir + i + ".jpg"):
                img_data.append(i+".jpg")
            elif os.path.exists(self.file_dir + i + ".JPG"):
                img_data.append(i+".JPG")
            elif os.path.exists(self.file_dir + i + ".png"):
                img_data.append(i+".png")
            elif os.path.exists(self.file_dir + i + ".PNG"):
                img_data.append(i+".PNG")
        print("data len : {}".format(len(img_data)))
        self.img_data = img_data

    def __getitem__(self, index):
        img_path = self.file_dir + self.img_data[index]

        with open(img_path, 'rb') as f:
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)

        bbox = np.array([[0,0,0,0,0]], dtype=np.float64)
        img, _ = self.transform((img, bbox))
        return img, None, None

    def __len__(self):
        return len(self.img_data)
