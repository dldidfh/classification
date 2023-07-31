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
def transfroms( dtype, img_size):
        if dtype == "train":
            return Compose([
                RandomResizedCrop((img_size,img_size)),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])
        elif dtype == "val" or dtype == "test" :
            return Compose([
                Resize((img_size,img_size)),
                # Resize((img_size,img_size), interpolation=Interpol),
                # CenterCrop((img_size,img_size)),
                ToTensor(),
                Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])
        else:
            raise  "check args - img size"