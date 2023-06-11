import os
import torch

from utils.SRutils import load_network
from super_resolution.rrdb import RRDBNet
import torchvision.transforms as transforms
from attacks.mdi2fgsm import get_norm_image, get_label, predict, mdi2fgsm

model_list = ["RRDBNet", ]
pretrained_weight_path = {"RRDBNet": './pretrained_model/PDM_Real_ESRGAN.pth'}

'''
      in_nc: 3
      out_nc: 3
      nf: 64
      nb: 23
      gc: 32
      upscale: 4
'''
def unormalize(x, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
    mean=torch.tensor(mean).view((1,-1,1,1))
    std=torch.tensor(std).view((1,-1,1,1))
    x=(x*std)+mean
    
    return torch.clip(x,0,None)

def main(model_name='RRDBNet'):
    # if model_name in model_list:
    #    model = model_name()

    # model 
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, upscale=4)
    load_network(model, pretrained_weight_path['RRDBNet'])

    img_name = 'ILSVRC2012_val_00000014_n04065272.JPEG'
    img_path = os.path.join('./dataset/benchmark', img_name) 

    img_path = './dataset/observation/adv/ILSVRC2012_val_00000003_n02105855.JPEG'
    img = get_norm_image(img_path)

    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    tensor2pil = transforms.ToPILImage()
    img_unorm = unormalize(img)

    output = model(img_unorm)

    img_sr_save = tensor2pil(output.cpu().clone().squeeze(0))
    img_sr_save.save(os.path.join('./dataset/observation/', img_name))



if __name__ == '__main__':
    main()