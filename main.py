import argparse
import sys, os 
import numpy as np 
from utils.check_utils import device_check
from models.call_model import load_model
from datas.data_loader import CustomLoader 
import evaluate
import torch 
from torch.utils.data import DataLoader
from models.steps import train, val
from datas.augmentations import transfroms
from pathlib import Path 
from datetime import datetime as dt 
def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    # dirs 
    parser.add_argument('--train_dir', type=str, default='dir/', help='train data folder')
    parser.add_argument('--val_dir', type=str, default='dir/', help='val data folder')
    parser.add_argument('--save_dir', type=str, default='dir/', help='model save dir path')
    
    # basics
    parser.add_argument('--device', type=str, default='0', help='cpu or 0,1,2...')

    # about datas 


    # about Model
    parser.add_argument('--model_type', type=str, default='timm', help='model hub - timm, hug, local, torch_hub')
    parser.add_argument('--model_path', type=str, default="timm/vit_tiny_patch16_224.augreg_in21k", help='model path')
    parser.add_argument('--img_size', type=int, default=224, help='core num workers')
    parser.add_argument('--num_cls', type=int, default=2, help='number of classes')
    parser.add_argument('--pretrained', action='store_true', help='call model with pretrained')


    # about training 
    parser.add_argument('--workers', type=int, default=0, help='core num workers')
    parser.add_argument('--epochs', type=int, default=3, help='epochs ')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate ')

    opt = parser.parse_args()
    return opt

def main(opt):
    p = Path(opt.save_dir)
    save_path = (p / dt.now().strftime("%y%m%d"))
    save_path.mkdir(parents=True, exist_ok=True)

    device = device_check(opt.device)
    model = load_model(opt.model_type, opt.model_path, device, opt.num_cls, opt.pretrained)

    # train_dataset = CustomLoader(root=opt.train_dir, img_size=(opt.img_size,opt.img_size), data_type="train")
    # val_dataset = CustomLoader(root=opt.val_dir, img_size=(opt.img_size,opt.img_size), data_type="val")
    from torchvision.datasets import ImageFolder
    # 왜 customloader로 바꾸면 데이터 전송이 안되나? 
    train_dataset = ImageFolder(root=opt.train_dir, transform=transfroms("train", opt.img_size))
    val_dataset = ImageFolder(root=opt.val_dir, transform=transfroms("val", opt.img_size))

    train_dataloder = DataLoader(train_dataset, batch_size=opt.batch_size, 
                                shuffle=True, num_workers=opt.workers)    
    val_dataloder = DataLoader(val_dataset, batch_size=opt.batch_size, 
                                shuffle=True, num_workers=opt.workers)   

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # min_loss = np.inf 
    min_acc = np.inf 

    for epoch in range(opt.epochs):
        train_loss, train_acc = train(opt, model, train_dataloder, loss_fn=loss_fn, device=device, optimizer=optimizer)
        val_loss, val_acc = val(opt, model, val_dataloder, loss_fn=loss_fn, device=device)

        if val_acc < min_acc : 
            print(f"[INFO] update loss {min_acc:.5f} to {val_loss:.5f}. saved")
            min_acc = val_acc
            model_save_path = os.path.join(save_path, f"epoch_{epoch}_f1_{min_acc*100:.0f}_loss_{val_loss:.3f}.pt")
            # 모델 저장 - 
            torch.save(model.state_dict(), model_save_path)
        print(f"epoch : {epoch}, train_loss : {train_loss:.3f}, train_f1 : {train_acc:.3f}, val_loss : {val_loss:.3f}, val_f1 : {val_acc:.3f}")
    
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
