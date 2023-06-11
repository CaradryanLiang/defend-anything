import torch

def unormalize(x, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
    mean=torch.tensor(mean).view((1,-1,1,1))
    std=torch.tensor(std).view((1,-1,1,1))
    x=(x*std)+mean
    
    return torch.clip(x,0,None)