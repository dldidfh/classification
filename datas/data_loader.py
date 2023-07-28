# from typing import Any, Callable, Optional
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

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
            

    