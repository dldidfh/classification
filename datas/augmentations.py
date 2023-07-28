# import albumemtations 
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

def transfroms(ttype, img_size):
    if ttype == "train":
        Compose(
        [
            RandomResizedCrop((img_size,img_size)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ]
    )
    elif ttype == "val":
        Compose(
        [
            Resize((img_size,img_size)),
            CenterCrop((img_size,img_size)),
            ToTensor(),
            Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ]
    )