# import albumemtations 
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,RandomVerticalFlip
)
import albumentations as A 
from albumentations.pytorch import ToTensorV2 as A_ToTensorV2
from albumentations.augmentations.geometric.resize import LongestMaxSize as A_LongestMaxSize
from albumentations.augmentations.geometric.transforms import Affine as A_Affine
from albumentations.augmentations.transforms import ColorJitter as A_ColorJitter

NORM_MEAN = 0.485, 0.456, 0.406  # RGB mean
NORM_STD = 0.229, 0.224, 0.225  # RGB standard deviation

def transfroms( dtype, img_size):
        if dtype == "train":
            return Compose([
                RandomResizedCrop((img_size,img_size)),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                ToTensor(),
                Normalize(mean=NORM_MEAN, std=NORM_STD),
            ])
        elif dtype == "val" or dtype == "test" :
            return Compose([
                Resize((img_size,img_size)),
                # Resize((img_size,img_size), interpolation=Interpol),
                # CenterCrop((img_size,img_size)),
                ToTensor(),
                Normalize(mean=NORM_MEAN, std=NORM_STD),
            ])
        else:
            raise  "check args - img size"
        
def albumentations_transforms( dtype, img_size):
     if dtype == "train":
          return A.Compose([
                # A_LongestMaxSize(img_size,always_apply=True),
                A.RandomResizedCrop(img_size, img_size),
                A_ColorJitter(hue=0.015,saturation=0.7, contrast=0.3, brightness=0.3),
                A_Affine(scale=1.0, p=0.1), # translate 
                # A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                # A.PadIfNeeded(img_size,img_size),
                A_ToTensorV2(), # albumentation.pytorch
          ])
     elif dtype == "val" or dtype == "test":
          return A.Compose([
                A_LongestMaxSize(img_size,always_apply=True),
                A.PadIfNeeded(img_size,img_size),
                A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                A_ToTensorV2(), # albumentation.pytorch
          ])
     else:
          raise "check args - img size"