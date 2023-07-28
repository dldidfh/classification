import timm 
import torch
# import huggingface_hub

def load_model(mtype:str, mpath:str, device:torch.device, ncls=1, pre=False): 
    if mtype == "timm":
        model = timm.create_model(mpath, pretrained=pre, num_classes=ncls)
    elif mtype == "torch_hub":
        model = torch.hub.load_state_dict_from_url(mpath)
    elif mtype == "hug":
        # model = huggingface_hub.load
        pass 
    elif mtype == "local":
        model = torch.load(mpath)
    else: 
        "check args - model type"
        "logging"
    model.to(device)
    return model 