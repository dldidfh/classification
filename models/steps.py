from torcheval.metrics.functional import multiclass_f1_score
from tqdm import tqdm
import numpy as np 
import torch 
TQDM_BAR_FORMAT = '{l_bar}{bar:30}{r_bar}'
def train(opt, model, dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0. 
    # corr = 0.
    # last_loss = 0. 
    label_list = []
    pred_list = []
    prograss_bar = tqdm(dataloader, desc="train : ", bar_format=TQDM_BAR_FORMAT)
    for img, label in prograss_bar:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        _, pred = output.max(dim=1)
        # corr += pred.eq(label).sum().item()
        for i, l in enumerate(label):
            pred_list.append(pred[i])
            label_list.append(l)
        running_loss += loss.item() * img.size(0)
    acc = multiclass_f1_score(torch.tensor(pred_list, dtype=torch.int64), torch.tensor(label_list, dtype=torch.int64), num_classes=opt.num_cls, average=opt.average)
    # acc = corr / len(dataloader.dataset)
    return running_loss / len(dataloader.dataset), acc 

def val(opt, model, dataloader, loss_fn, device):
    model.eval()
    prograss_bar = tqdm(dataloader, desc="val : ", bar_format=TQDM_BAR_FORMAT)
    with torch.no_grad():
        corr = 0. 
        running_loss = 0. 
        label_list = []
        pred_list = []
        for img, label in prograss_bar:
            img, label = img.to(device), label.to(device)
            output = model(img)
            _, pred = output.max(dim=1)
            # corr += torch.sum(pred.eq(label)).item()
            for i, l in enumerate(label):
                pred_list.append(pred[i])
                label_list.append(l)

            running_loss += loss_fn(output, label).item() * img.size(0)
        acc = multiclass_f1_score(torch.tensor(pred_list, dtype=torch.int64), torch.tensor(label_list, dtype=torch.int64), num_classes=opt.num_cls, average=opt.average)        # acc = corr / len(dataloader.dataset)
    return running_loss / len(dataloader.dataset), acc 
    
def test(opt, model, dataloader, device):
    model.eval()
    prograss_bar = tqdm(dataloader, desc="test :")
    with torch.no_grad():
        for img, label in prograss_bar:
            img, label = img.to(device), label.to(device)
            output = model(img)
            _, pred = output.max(dim=1)
    return 
        