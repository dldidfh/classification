import argparse
import logging 
import transformers
import sys, os 
import numpy as np 
import timm 
from utils.check_utils import device_check
from models.call_model import load_model
import evaluate
import torch 
from tqdm import tqdm 
from torch.utils.data import DataLoader, random_split
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
def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    # dirs 
    parser.add_argument('--train_dir', type=str, default='dir/', help='train data folder')
    parser.add_argument('--val_dir', type=str, default='dir/', help='val data folder')
    parser.add_argument('--save_dir', type=str, default='dir/', help='model save dir path')
    
    # basics
    parser.add_argument('--device', type=str, default='0', help='cpu or 0,1,2...')

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
    # Load the accuracy metric from the datasets package
    device = device_check(opt.device)
    model = load_model(opt.model_type, opt.model_path, device, opt.num_cls, opt.pretrained)
    



    _train_transforms = Compose(
        [
            RandomResizedCrop((opt.img_size,opt.img_size)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ]
    )
    _val_transforms = Compose(
        [
            Resize((opt.img_size,opt.img_size)),
            CenterCrop((opt.img_size,opt.img_size)),
            ToTensor(),
            Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ]
    )
    train_dataset = ImageFolder(root=opt.train_dir,
                                transform=_train_transforms)
    val_dataset = ImageFolder(root=opt.val_dir,
                               transform=_val_transforms)
    
    train_dataloder = DataLoader(train_dataset, 
               batch_size=opt.batch_size, 
               shuffle=True, 
               num_workers=opt.workers)    
    val_dataloder = DataLoader(val_dataset, 
               batch_size=opt.batch_size, 
               shuffle=True, 
               num_workers=opt.workers)   
    

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.to(device)
    # optimizer.to(device)

    def train(model, dataloader, loss_fn, optimizer, device):
        model.train()
        running_loss = 0. 
        corr = 0.
        # last_loss = 0. 
        prograss_bar = tqdm(dataloader, desc="train")
        for img, label in prograss_bar:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            _, pred = output.max(dim=1)
            corr += pred.eq(label).sum().item()
            running_loss += loss.item() * img.size(0)
        acc = corr / len(dataloader.dataset)
        return running_loss / len(dataloader.dataset), acc 
    def val(model, dataloader, loss_fn, device):
        model.eval()
        prograss_bar = tqdm(dataloader, desc="val")
        with torch.no_grad():
            corr = 0. 
            running_loss = 0. 
            for img, label in prograss_bar:
                img, label = img.to(device), label.to(device)
                output = model(img)
                _, pred = output.max(dim=1)
                corr += torch.sum(pred.eq(label)).item()

                running_loss += loss_fn(output, label).item() * img.size(0)
            acc = corr / len(dataloader.dataset)
        return running_loss / len(dataloader.dataset), acc 
    

    min_loss = np.inf 
    for epoch in range(opt.epochs):
        train_loss, train_acc = train(model, train_dataloder, loss_fn=loss_fn, device=device, optimizer=optimizer)
        val_loss, val_acc = val(model, train_dataloder, loss_fn=loss_fn, device=device)

        if val_loss < min_loss : 
            print(f"[INFO] update loss {min_loss:.5f} to {val_loss:.5f}. saved")
            min_loss = val_loss
            model_save_path = os.path.join(opt.save_dir, f"epoch_{epoch}_loss_{val_loss:.2f}.pt")
            torch.save(model.state_dict(), model_save_path)
        print(f"epoch : {epoch}, loss : {train_loss:.3f}, acc : {train_acc:.2f}, val_loss : {val_loss:.3f}, val_acc : {val_acc:.2f}")

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
