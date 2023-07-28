import os 
import torch 


def device_check(device:str):
    if device == "cpu" or device.startswith("c"):
        device = "cpu"
    elif device:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.split(",")), f"check args - device"
        device = "cuda:0"
    return torch.device(device)
