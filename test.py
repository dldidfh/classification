import os 
import numpy as np 
# image_filepath = "test_data/test_dir2/test.jpg"
# a = os.path.normpath(image_filepath).split(os.sep)[-2]
# print(a)


# a = np.array([1,2,3,4,5])
# b = np.array([0,0,0,0,0])
# c = np.stack((a,b), axis=-1)
# # c = np.concatenate((a,b), axis=-1)
# for i in c:
#     print(i)

from datas.data_loader import CustomDataSet
from datas.augmentations import albumentations_transforms
# a = CustomDataSet("test_data/test_dir2",'ram',transform=albumentations_transforms(dtype="train",img_size=224))
# for img, label in a:
#     print(111)

from torch.nn import CrossEntropyLoss, BCELoss
import torch 
# a = torch.tensor([0,1,2])
# a = torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
# b = torch.tensor([[0.2,0.1,0.1],[0.5,0.1,0.4], [1.,0.3,0.]])
a = torch.tensor([[1.,0.],[1.,0.],[0.,0.]])
b = torch.tensor([[0.2,0.1],[0.1,0.4], [0.3,1.]])
print(a.size(), b.size())
# loss_fn = CrossEntropyLoss()
loss_fn = BCELoss()
# c = loss_fn(a,b)
c = loss_fn(b,a)
print(c)