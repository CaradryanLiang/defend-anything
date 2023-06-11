import os
import json
import re

with open('./dataset/imagenet_class_mapping.json') as file:
    mapping = json.load(file)

def get_label(img_name: str):
    pattern = r'n\d*'
    cls = re.search(pattern, img_name).group()
    label_num = mapping['cls2idx'][cls]
    print(label_num)

def main():
    img_name = 'ILSVRC2012_val_00000014_n04065272.JPEG'
    get_label(img_name)
    return

if __name__ == '__main__':
    main()