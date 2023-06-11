import os
import torch
import PIL.Image as Image
import json


from attack.attack import get_norm_image, get_label, predict, mdi2fgsm, pgd, deepfool, fgsm
from torchvision.models import inception_v3
import torchvision.transforms as transforms
from defend_anything.defend_anything import DefendAnythingWrapper
from utils.common import unormalize
from utils.option import parse


def eval(img_name, model, adv_model, defense):
    # img_name = 'ILSVRC2012_val_00000014_n04065272.JPEG'
    img_path = os.path.join('./dataset/benchmark', img_name) 
    label = get_label(img_name)
    img = get_norm_image(img_path)

    # adv_image = fgsm(img, label, adv_model)
    adv_image = mdi2fgsm(img, label, adv_model)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    tensor2pil = transforms.ToPILImage()
    adv_img_unorm = unormalize(adv_image)

    adv_image_save = tensor2pil(adv_img_unorm.cpu().clone().squeeze(0))
    adv_image_save.save(os.path.join('./dataset/observation/adv', img_name))

    img_denoise_da_unorm = defense.run(adv_img_unorm)
    img_denoise_da = normalize(img_denoise_da_unorm)
    img_da_save = tensor2pil(img_denoise_da_unorm.cpu().clone().squeeze(0))
    img_da_save.save(os.path.join('./dataset/observation/da', img_name))

    # prediction = predict(img, model)
    adv_prediction = predict(adv_image, model)
    da_prediction = predict(img_denoise_da, model)

    # print(prediction, adv_prediction)

    return label, adv_prediction, da_prediction, label

def main(opt):
     # Step 1: Initialize model with the best available weights
    model = inception_v3(pretrained=True)
    adv_model = inception_v3(pretrained=True)
    model.eval()

    da = DefendAnythingWrapper(opt)

    img_cnt = 0
    correct_cnt = 0
    adv_correct_cnt = 0
    da_correct_cnt = 0

    for img_name in os.listdir('./dataset/benchmark/'):
        pred, adv_pred, da_pred, label = eval(img_name, model, adv_model, da)
        
        img_cnt = img_cnt + 1
        if adv_pred == label:
            adv_correct_cnt = adv_correct_cnt + 1
        if pred == label:
            correct_cnt = correct_cnt + 1
        if da_pred == label:
            da_correct_cnt = da_correct_cnt + 1
        
        acc = float(correct_cnt)/float(img_cnt)
        adv_acc = float(adv_correct_cnt)/float(img_cnt)
        da_acc = float(da_correct_cnt)/float(img_cnt)

        print("image id: {}, label: {}, pred: {}, adv_pred: {}, da_pred: {}, acc: {}, adv_acc: {}, da_acc: {}".format(img_cnt, label, pred, adv_pred, da_pred, acc, adv_acc, da_acc))

    
    print("Final Result>>>  image number: {}, correct/acc: {}/{}, adv correct/acc: {}/{}".format(img_cnt, correct_cnt, acc, adv_correct_cnt, adv_acc))
    result = (img_cnt, correct_cnt, acc, adv_correct_cnt, adv_acc)
    with open("./result.json", 'w') as file:
        json.dump(result, file)



if __name__ == '__main__':
    opt_path = 'defend_anything/config/defend-anything-standard.yml'
    opt = parse(opt_path=opt_path)
    torch.manual_seed(opt['seed'])
    main(opt)
