from typing import Any, Callable, Optional
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from datas.augmentations import transfroms 

class CustomLoader(ImageFolder):
    def __init__(self, root: str, img_size=(224,224) ,data_type="train"):
        super().__init__(root=root)
        self.transforms = transfroms(data_type, img_size)
        

    