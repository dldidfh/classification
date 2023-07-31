from glob import glob 
import torch 
from pathlib import Path
import argparse
import cv2 
import os 
import shutil
from tqdm import tqdm 
import timm 
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Resize, Compose, ToPILImage
from torchvision.datasets import ImageFolder
from datas.augmentations import transfroms
from models.steps import test 
from utils.check_utils import device_check
from PIL import Image
IMG_EXT = ["jpg", "png", "bmp", "jpeg"]
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="engines/dddd.pt", help='model path')
    parser.add_argument('--img_dir', type=str, default='dir/', help='test image dir path')
    parser.add_argument('--save_dir', type=str, default='dir/', help='result save dir path')
    parser.add_argument('--cls_num', type=int, default=2, help='num of classes')
    parser.add_argument('--img_size', type=int, default=224, help='model input image size')
    parser.add_argument('--batch_size', type=int, default=1, help='inference batch size')
    parser.add_argument('--workers', type=int, default=0, help='core num workers')
    parser.add_argument('--device', type=str, default='0', help='cpu or 0,1,2...')


    opt = parser.parse_args()
    return opt

def main(opt):
    device = device_check(opt.device)
    files = []
    # 모델 로드
    model = timm.create_model("timm/vit_tiny_patch16_224.augreg_in21k", num_classes=opt.cls_num)
    # for i, param in enumerate(model.parameters()):
    #     if i == 0 :
    #         print(param)
    model.load_state_dict(torch.load(opt.model_path))


    # 클레스 별 폴더 생성
    for i in range(opt.cls_num):
        Path(os.path.join(opt.save_dir, str(i))).mkdir(exist_ok=True, parents=True)
    # 이미지 불러오기
    for ext in IMG_EXT:
        files += glob(os.path.join(opt.img_dir, f"*.{ext}"))
    transform = transfroms("test", opt.img_size)
    
    model.to(device)
    with torch.no_grad():
        for f in tqdm(files, desc="classification :"):
            img = Image.open(f)
            img = transform(img)
            if len(img) == 3 :
                img = img[None]
            img = img.to(device)
            output = model(img)
            score, pred = output.max(dim=1)
            # 이미지 분류 - 이동 또는 복제
            f_base_name = os.path.basename(f)
            shutil.copy(f, os.path.join(opt.save_dir, str(int(pred)), f"{float(score):.3f}_" + f_base_name))
            # shutil.move(f, os.path.join(opt.save_dir, pred, f_base_name))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
