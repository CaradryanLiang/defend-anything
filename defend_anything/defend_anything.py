import sys 
sys.path.append("..")

import torch.nn as nn
import torch
import random
import datetime
import PIL.Image as Image

from torchvision import transforms
from skimage.restoration import denoise_wavelet
import torch.nn.functional as F


from super_resolution.edsr import EDSR
from super_resolution.rrdb import RRDBNet
from utils.SRutils import load_network


class WaveletDenoising(nn.Module):
    def __init__(self):
        super(WaveletDenoising, self).__init__()
        return
    
    def forward(self, x):
        x_array = x.permute(0, 2, 3, 1).detach().numpy()
        x_bayes = denoise_wavelet(x_array, convert2ycbcr=True,
                               method='BayesShrink', mode='soft',sigma=0.01, channel_axis=3)
        return torch.from_numpy(x_bayes).permute(0, 3, 1, 2)
    
    def backward(self, grad_downstream):
        return grad_downstream
    
class DABlock(nn.Module):
    def __init__(self, sr_model, scale):
        super(DABlock, self).__init__()
        self.denoiser = WaveletDenoising()
        self.downsampler8 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.downsampler4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.downsampler2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.sr_model = sr_model

    
    def forward(self, x):
        x = self.denoiser(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.denoiser(x)
        x = self.downsampler4(x)
        x = self.denoiser(x)
        x = self.sr_model(x)
        x = self.denoiser(x)
        return x
    
class DefendAnything(nn.Module):
    def __init__(self, opt):
        '''
            model_num: total number of used model (one model can forward for several times)
        
        '''
        super(DefendAnything, self).__init__()
        self.opt = opt
        self.max_length = opt['max_length']

        self.block_dict = {}
        for name in opt['model_list']:
            assert name in opt.keys()
            
            # build sr_net
            sr_net = RRDBNet(**opt[name]['args'])
            load_network(sr_net, opt[name]['pretrained_path'])
            
            # build defend anything block
            scale = opt[name]['scale']
            da_block = DABlock(sr_net, scale)

            self.block_dict[name] = da_block



    def forward(self, x):
        random.seed(datetime.datetime.now())
        length = random.randint(1, self.max_length)
        x_list = [random.randint(0, len(self.block_dict)-1) for _ in range(length)]

        sr_key_list = [list(self.block_dict.keys())[i] for i in x_list]

        for key in sr_key_list:
            model = self.block_dict[key]
            x = model(x)

        return x

        

class DefendAnythingWrapper():
    '''
        Do:
            initialize the model list for DefendAnything

    '''
    def __init__(self, opt):
        # initialize model
        self.model = DefendAnything(opt)

    def load_image_as_tensor(self, img_path):
        with open(img_path, 'r') as file:
            img = Image.open(file)
            if len(img.split()) < 3:
                img = img.convert('RGB')

        transform = transforms.Compose([transforms.ToTensor(),]) 
        img = transform(img)
        img = torch.unsqueeze(0)

        return img
    
    def run(self, img):
        x = self.model(img)
        return x

            

    
