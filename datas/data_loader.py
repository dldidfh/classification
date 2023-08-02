# from typing import Any, Callable, Optional
from glob import glob 
import os 
import numpy as np 
import random
import cv2 
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
EXT_LIST = ["jpg", "png", "bmp", "jpeg"]
class CustomLoader(ImageFolder):
    def __init__(self, root: str, img_size=(224,224) ,data_type="train"):
        transforms = self.transfroms(data_type, img_size)
        super().__init__(root=root, transform=transforms)
        
    def transfroms(self, dtype, img_size):
        if dtype == "train":
            return Compose([
                RandomResizedCrop((img_size,img_size)),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])
        elif dtype == "val":
            return Compose([
                Resize((img_size,img_size)),
                CenterCrop((img_size,img_size)),
                ToTensor(),
                Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])
        else:
            raise  "check args - img size"
            
class CustomDataSet(Dataset):
    def __init__(self, root_dir, cls_num, cache='ram',transform=None):
        self.root_dir = root_dir
        self.cls_num = cls_num
        self.cache = cache
        self.transform = transform
        self.datas = []
        self.load_datasets()

    def load_datasets(self):
        classes_dir = os.listdir(self.root_dir)
        for i, cls_name in enumerate(classes_dir):
            if os.path.isdir(os.path.join(self.root_dir, cls_name)):
                image_path_list_str = []
                image_path_list = []
                for ext in EXT_LIST:
                    for f in glob(f"{os.path.join(self.root_dir, cls_name)}/*.{ext}"):
                        # temp = np.zeros(self.cls_num,dtype=np.float32)
                        # temp[i] = 1 
                        image_path_list_str.append((f, i)) # 레이블이 굳이 class 개수만큼의 범위를 가질 필요 없음

                if self.cache == "ram":
                    for img, label in image_path_list_str:
                        img = cv2.imread(img)
                        image_path_list.append( (img,label))
                else:
                    image_path_list = image_path_list_str

            self.datas += image_path_list
        random.shuffle(self.datas)

    def __getitem__(self, index):
        if self.cache == 'ram':
            if self.transform is not None: 
                img = self.datas[index][0]
                img = self.transform(image=img)['image']
            return img, self.datas[index][1]
        else:
            img = cv2.imread(self.datas[index][0])
            if self.transform is not None: 
                img = self.datas[index][0]
                img = self.transform(image=img)['image']
            return img, self.datas[index][1]


    def __len__(self):
        return len(self.datas)

    