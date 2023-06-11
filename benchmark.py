import os
import torch
import torch.nn as nn
import json
import PIL.Image as Image
import re
import torch.hub

from shutil import copyfile

import torchattacks
from torchvision.models import inception_v3
from torchvision.io import read_image
import torchvision.transforms as transforms

with open('./dataset/sample_list.json') as file:
    sample_list = json.load(file)

for img_name in sample_list:
    img_path = os.path.join('./dataset/val_images/', img_name)
    copyfile(img_path, os.path.join('./dataset/benchmark/', img_name))
    

