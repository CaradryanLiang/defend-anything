import os
import torch
import torch.nn as nn
import json
import PIL.Image as Image
import re
import torch.hub

import torchattacks
from torchvision.models import inception_v3
from torchvision.io import read_image
import torchvision.transforms as transforms

with open('./dataset/imagenet_class_mapping.json') as file:
    mapping = json.load(file)

def get_label(img_name: str):
    pattern = r'n\d*'
    cls = re.search(pattern, img_name).group()
    label_num = mapping['cls2idx'][cls]

    label = torch.zeros((1,), dtype=torch.long)
    label[0] = label_num

    return label
    

# refer to the guideline in torchvision: https://pytorch.org/vision/stable/models.html
def get_norm_image(img_path: str):
    # Step 0: load image
    with open(img_path, 'rb') as file:
        img = Image.open(file)
        if len(img.split()) < 3:
            img = img.convert('RGB')


        preprocess = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(256), 
                                        transforms.CenterCrop(224),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),      
                                    ])
        img = preprocess(img).unsqueeze(0)

    return img


def mdi2fgsm(img, label, model):
    '''
        img: [1, C, H, W]
        label: [1]
    '''
    attack = torchattacks.DIFGSM(model, eps=8/255, alpha=2/255, steps=4)
    # If, images are normalized:
    attack.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    adv_img = attack(img, label)

    return adv_img

def pgd(img, label, model):
    '''
        img: [1, C, H, W]
        label: [1]
    '''
    attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)
    # If, images are normalized:
    attack.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    adv_img = attack(img, label)

    return adv_img

def fgsm(img, label, model):
    '''
        img: [1, C, H, W]
        label: [1]
    '''
    attack = torchattacks.FGSM(model, eps=2/255)
    # If, images are normalized:
    attack.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    adv_img = attack(img, label)

    return adv_img

def deepfool(img, label, model):
    '''
        img: [1, C, H, W]
        label: [1]
    '''
    attack = torchattacks.DeepFool(model)
    # If, images are normalized:
    attack.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    adv_img = attack(img, label)

    return adv_img

def predict(img, model):


    # Step 4: Use the model and print the predicted category
    prediction = model(img).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    # print(f"{class_id}: {100 * score:.1f}%")

    return class_id


def eval(img_name, model, adv_model):
    # img_name = 'ILSVRC2012_val_00000014_n04065272.JPEG'
    img_path = os.path.join('./dataset/val_images', img_name) 
    label = get_label(img_name)
    img = get_norm_image(img_path)

    prediction = predict(img, model)
    adv_prediction = predict(mdi2fgsm(img, label, adv_model), model)

    # print(prediction, adv_prediction)

    return prediction, adv_prediction, label

def main():
     # Step 1: Initialize model with the best available weights
    model = inception_v3(pretrained=True)
    adv_model = inception_v3(pretrained=True)
    model.eval()

    img_cnt = 0
    correct_cnt = 0
    adv_correct_cnt = 0
    sample_list = []

    for img_name in os.listdir('./dataset/val_images/'):
        pred, adv_pred, label = eval(img_name, model, adv_model)
        
        img_cnt = img_cnt + 1
        if adv_pred == label:
            adv_correct_cnt = adv_correct_cnt + 1
        if pred == label:
            correct_cnt = correct_cnt + 1
            if len(sample_list) < 5000:
                sample_list.append(img_name)
        
        acc = float(correct_cnt)/float(img_cnt)
        adv_acc = float(adv_correct_cnt)/float(img_cnt)

        print("image id: {}, label: {}, pred: {}, adv_pred: {}, acc: {}, adv_acc: {}".format(img_cnt, label, pred, adv_pred, acc, adv_acc))

        if len(sample_list) >= 5000:
            break

    
    print("Final Result>>>  image number: {}, correct/acc: {}/{}, adv correct/acc: {}/{}".format(img_cnt, correct_cnt, acc, adv_correct_cnt, adv_acc))
    with open("./sample_list.json", 'w') as file:
        json.dump(sample_list, file)

    




if __name__ == '__main__':
    main()

