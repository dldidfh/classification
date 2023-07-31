from torcheval.metrics.functional import multiclass_f1_score
from tqdm import tqdm
import torch 

def train(opt, model, dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0. 
    # corr = 0.
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
        # corr += pred.eq(label).sum().item()
        running_loss += loss.item() * img.size(0)
    acc = multiclass_f1_score(pred, label, num_classes=opt.num_cls)
    # acc = corr / len(dataloader.dataset)
    return running_loss / len(dataloader.dataset), acc 

def val(opt, model, dataloader, loss_fn, device):
    model.eval()
    prograss_bar = tqdm(dataloader, desc="val")
    with torch.no_grad():
        corr = 0. 
        running_loss = 0. 
        for img, label in prograss_bar:
            img, label = img.to(device), label.to(device)
            output = model(img)
            _, pred = output.max(dim=1)
            # corr += torch.sum(pred.eq(label)).item()

            running_loss += loss_fn(output, label).item() * img.size(0)
        acc = multiclass_f1_score(pred, label, num_classes=opt.num_cls)
        # acc = corr / len(dataloader.dataset)
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
        