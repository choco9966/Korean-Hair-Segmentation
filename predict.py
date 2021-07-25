import json
import pandas as pd
import sys
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import os
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import albumentations as A
import argparse    
from tqdm import tqdm
import torch.nn.functional as F
import datetime
import cv2
from PIL import Image
import numpy as np
def get_args():
    parser = argparse.ArgumentParser(description='Hair Segmentation')
    parser.add_argument('--testset_path', type=str, default='testset_path') # korean, celeb
    parser.add_argument('--encoder', type=str, default='encoder')
    parser.add_argument('--decoder', type=str, default='decoder')
    args = parser.parse_args()
    return args

# args = get_args()

testset_path = './DATA/Final_DATA/task02_test'
encoder = 'timm-efficientnet-b2'
decoder = 'UnetPlusPlus'



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
def get_transforms(*, data):
    if data == 'train':
        return A.Compose([
                A.Resize(512, 512,always_apply=True),
                A.OneOf([
                    A.RandomContrast(),
                    A.RandomGamma(),
                    A.RandomBrightness(),
                    ], p=0.3),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    A.GridDistortion(),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                    ], p=0.3),
                A.ShiftScaleRotate(p=0.2),
                A.GridDropout(p=0.1), 
                A.Resize(512,512,always_apply=True),
                A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(transpose_mask=True)
        ],p=1.)
    elif data == 'valid':
        return A.Compose([
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(transpose_mask=True)
        ],p=1.)



class CFG:
    encoder_type= encoder
    decoder_type= decoder
    num_workers = 0
class Encoder(nn.Module):
    def __init__(self, encoder_name='timm-efficientnet-b3', decoder_name='Unet' , pretrained=False):
        super().__init__()
        if CFG.encoder_type in ['se_resnext50_32x4d', 'se_resnext101_32x4d']: 
            encoder_weights = 'imagenet' 
        else: 
            encoder_weights = 'noisy-student' 
        
        if CFG.decoder_type == 'Unet': 
            self.encoder = smp.Unet(encoder_name=CFG.encoder_type, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif CFG.decoder_type == 'UnetPlusPlus':
            self.encoder = smp.UnetPlusPlus(encoder_name=CFG.encoder_type, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif CFG.decoder_type == 'MAnet': 
            self.encoder = smp.MAnet(encoder_name=CFG.encoder_type, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif CFG.decoder_type == 'Linknet': 
            self.encoder = smp.Linknet(encoder_name=CFG.encoder_type, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif CFG.decoder_type == 'FPN':
            self.encoder = smp.FPN(encoder_name=CFG.encoder_type, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif CFG.decoder_type == 'PSPNet': 
            self.encoder = smp.PSPNet(encoder_name=CFG.encoder_type, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif CFG.decoder_type == 'PAN': 
            self.encoder = smp.PAN(encoder_name=CFG.encoder_type, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif CFG.decoder_type == 'DeepLabV3': 
            self.encoder = smp.DeepLabV3(encoder_name=CFG.encoder_type, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif CFG.decoder_type == 'DeepLabV3Plus': 
            self.encoder = smp.DeepLabV3Plus(encoder_name=CFG.encoder_type, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        else:
            raise ValueError(f"decoder_type : {CFG.decoder_type} is not exist")
           
        
    #@autocast()
    def forward(self, x):
        x = self.encoder(x)
        return x
    
    

    

test_df = pd.DataFrame()
test_df['images'] = [testset_path + '/images/' + c.split('.')[0] + '.jpg' for c in sorted(os.listdir(testset_path + '/images'))]
test_df['ids'] = test_df['images'].apply(lambda x: x.split('/')[-1])

class TestDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
        
    def __getitem__(self, idx):
        ids = self.data.loc[idx]['ids']
        images = cv2.imread(self.data.loc[idx]['images'])    
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=images)
            images = transformed["image"]

        return ids, images

    def __len__(self):
        return len(self.data)

# ====================================================
# loader
# ====================================================
test_dataset = TestDataset(test_df, transform=get_transforms(data='valid'))


test_loader = DataLoader(test_dataset, 
                          batch_size=32, 
                          shuffle=False, 
                          num_workers=CFG.num_workers,
                          pin_memory=True, 
                          drop_last=False)

collect = {"annotations":[]}

def mergeContours(cnt1, cnt2):
    minDist1 = np.inf
    minInd1 = None
    for ind, point in enumerate(cnt1):
        dist = cv2.pointPolygonTest(cnt2, (int(point[0][0]), int(point[0][1])), True)
        dist = abs(dist)
        if dist < minDist1:
            minDist1 = dist
            minInd1 = ind
    cnt1 = np.roll(cnt1, -minInd1, axis=0)

    minDist2 = np.inf
    minInd2 = None
    for ind, point in enumerate(cnt2):
        dist = cv2.pointPolygonTest(cnt1, (int(point[0][0]), int(point[0][1])), True)
        dist = abs(dist)
        if dist < minDist2:
            minDist2 = dist
            minInd2 = ind
    cnt2 = np.roll(cnt2, -minInd2, axis=0)
    mreged_cnt = np.concatenate([cnt1,cnt2],axis=0)
    return mreged_cnt

import ttach as tta
transforms = tta.Compose(
    [
        tta.HorizontalFlip()    
    ]
)

model_path = f'./output/runs/timm-efficientnet-b2_UnetPlusPlus_fold0___hair_66%_image512_iou80_pixel12_DiceBCE_cutmixColor01_epoch13.pth'
encoder1 = Encoder(CFG.encoder_type, CFG.decoder_type, pretrained=False)
checkpoint = torch.load(model_path, map_location=device)
encoder1.load_state_dict(checkpoint['encoder'])
encoder1.to(device)
model_path = f'./output/runs/timm-efficientnet-b2_UnetPlusPlus_fold0___hair_66%_image512_iou80_pixel12_DiceBCE_cutmixColor01_epoch11.pth'
encoder2 = Encoder(CFG.encoder_type, CFG.decoder_type, pretrained=False)
checkpoint = torch.load(model_path, map_location=device)
encoder2.load_state_dict(checkpoint['encoder'])
encoder2.to(device)
model_path = f'./output/runs/timm-efficientnet-b2_UnetPlusPlus_fold0___hair_66%_image512_iou80_pixel12_DiceBCE_cutmixColor01_epoch12.pth'
encoder3 = Encoder(CFG.encoder_type, CFG.decoder_type, pretrained=False)
checkpoint = torch.load(model_path, map_location=device)
encoder3.load_state_dict(checkpoint['encoder'])
encoder3.to(device)
tta_model1 = tta.SegmentationTTAWrapper(encoder1, transforms)
tta_model2 = tta.SegmentationTTAWrapper(encoder2, transforms)
tta_model3 = tta.SegmentationTTAWrapper(encoder3, transforms)
tta_model1.eval()
encoder1.eval()
tta_model2.eval()
encoder2.eval()
tta_model3.eval()
encoder3.eval()
for step, (ids, images) in enumerate(tqdm(test_loader)):
    images = images.to(device)
    with torch.no_grad():
        y_preds = (1/6)*(encoder1(images)+tta_model1(images)+encoder2(images)+tta_model2(images)
                            + encoder3(images)+tta_model3(images))   

    y_preds = F.sigmoid(y_preds)
    y_preds = y_preds > 0.5

    # n_mask = torch.sum(y_preds > 0.5, dim=(1,2,3))
    # mask_threshold = 14000
    # small_mask_ind = torch.where(n_mask < mask_threshold)
    # y_preds[small_mask_ind] = torch.where(y_preds[small_mask_ind] > 0.55, 1.0, 0.0)
    # y_preds = torch.where(y_preds > 0.49, 1.0, 0.0)

    y_preds = y_preds.detach().cpu().numpy()
    y_preds = np.uint8(y_preds*255)

    for idx, y_pred in enumerate(y_preds):
        y_pred = y_pred.transpose(1, 2, 0)
        contours, hierarchy = cv2.findContours(y_pred, cv2.RETR_LIST ,cv2.CHAIN_APPROX_NONE)    
        contours = sorted(contours, key= lambda x: cv2.contourArea(x), reverse=True)
        if len(contours) == 0:
            cnt = np.array([[[0,0]],[[0,0]],[[0,0]]])
        else:
            cnt = contours[0]
        for i in range(1,len(contours)):
            if cv2.contourArea(contours[i]) > 75:
                cnt = mergeContours(cnt, contours[i])
        # 적용하는 숫자가 커질 수록 Point의 갯수는 감소
        epsilon = 0.555
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 50:
            polygons = cnt
        else:
            polygons = approx
        x_and_y = [{"x":int(a[0][0]), "y": int(a[0][1])} for a in polygons]
        collect["annotations"].append({"file_name" : ids[idx],
                                        "polygon1" : x_and_y})
with open('result.json','w') as json_file:
    json.dump(collect, json_file)