import os
import json

import cv2
import numpy as np
import pandas as pd
import torch
from shapely.geometry import Polygon
from scipy import stats
from tqdm import tqdm
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from modules.utils import load_yaml
from modules.dataset import ValidDataset
from modules.transform import get_transforms_preprocessing
from modules.scheduler import get_dice_coeff
from modules.models import Encoder

CONFIG_PATH = './config/config.yaml'
config = load_yaml(CONFIG_PATH)

# PREPROCESSING
ENCODER_TYPE = config['PREPROCESSING']['encoder_type']
DECODER_TYPE = config['PREPROCESSING']['decoder_type']
PRE_N_FOLD = config['PREPROCESSING']['n_fold']
PRE_TRN_FOLD = config['PREPROCESSING']['trn_fold']
NPIXEL_THRESHOLD = config['PREPROCESSING']['npixel_threshold']
NPIXEL_FOR_IOU = config['PREPROCESSING']['npixel_for_iou']
IOU_THRESHOLD = config['PREPROCESSING']['iou_threshold']

# SEED
RANDOM_SEED = config['SEED']['random_seed']

# DATALOADER
NUM_WORKERS = config['DATALOADER']['num_workers']

# LOG
VERSION = config['LOG']['version']

def make_masks(dataset_path):
    if not os.path.exists(dataset_path+'/masks'):
        os.makedirs(dataset_path+'/masks')

    with open(dataset_path+"/labels.json", "r") as l:
        data = json.load(l)

    entries = {}
    for idx, files in enumerate(data['annotations']):
        entries[files['file_name']]=[]
        for polygon in files['polygon1']:
            entries[files['file_name']].append(tuple(polygon.values()))

    width = height = 512
    for name, data in tqdm(entries.items()):
        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(data, outline=1, fill=1)
        img.save(os.path.join(dataset_path+'/masks',os.path.splitext(name)[0])+'.png')
    
    
def make_pre_data():
    data_pre = pd.read_csv('./DATA/data.csv')
    data_pre = data_pre.sort_values(by='npixels', ascending=True)
    data_pre = data_pre.reset_index(drop=True)
    data_pre = data_pre[data_pre['npixels'] > NPIXEL_FOR_IOU].reset_index(drop=True)

    data_pre['npixels_bins'] = pd.qcut(data_pre['npixels'], q=10, retbins=True, labels=False)[0].values
    skf = StratifiedKFold(PRE_N_FOLD, random_state = RANDOM_SEED, shuffle=True)

    data_pre['fold'] = 0
    for fold, (tr_idx, val_idx) in enumerate(skf.split(data_pre, y=data_pre['npixels_bins'])):
        data_pre.loc[val_idx, 'fold'] = fold

    return data_pre

   
def make_npixels_data(dataset_path):
    data_npixels = pd.DataFrame()
    data_npixels['images'] = [dataset_path + '/images/' + c.split('.')[0] + '.jpg' for c in sorted(os.listdir(dataset_path + '/masks'))]
    data_npixels['masks'] = [dataset_path + '/masks/' + c.split('.')[0] + '.png' for c in sorted(os.listdir(dataset_path + '/masks'))]
    data_npixels['npixels'] = data_npixels['masks'].apply(lambda x: np.count_nonzero(cv2.imread(x))/3)
    # data_npixels = data_npixels.sort_values(by='npixels', ascending=True)
    data_npixels.to_csv('./DATA/data.csv', index=False)


def make_data():
    data = pd.read_csv('./DATA/data.csv')
    data = data.sort_values(by='npixels',ascending=True)
    data = data.loc[data.npixels >= NPIXEL_THRESHOLD]
    #data = data.iloc[20000:-13000]    
    data = data.reset_index(drop=True)
    #data['images'] = data['images'].str.replace('data/train','DATA/Final_DATA/task02_train')
    #data['masks'] = data['masks'].str.replace('data/train','DATA/Final_DATA/task02_train')
    
    return data

def calculate_iou(data, device):
    target_dir = './DATA/iou_result/preds'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    image_ids = []
    dice_score = []
    for fold in PRE_TRN_FOLD:
        valids = data[data['fold'] == fold].reset_index(drop=True)
        valid_dataset = ValidDataset(valids, transform=get_transforms_preprocessing(data='valid'))
        valid_loader = DataLoader(valid_dataset, 
                                  batch_size=128, 
                                  shuffle=False, 
                                  num_workers=NUM_WORKERS,
                                  pin_memory=True, 
                                  drop_last=False)    

        model_path1 = f'./model/preprocessing/{ENCODER_TYPE}_{DECODER_TYPE}_fold{fold}_{VERSION}_1.pth'
        encoder1 = Encoder(ENCODER_TYPE, DECODER_TYPE, pretrained=False)

        checkpoint = torch.load(model_path1, map_location=device)
        encoder1.load_state_dict(checkpoint['encoder'])
        encoder1.to(device)

        model_path2 = f'./model/preprocessing/{ENCODER_TYPE}_{DECODER_TYPE}_fold{fold}_{VERSION}_2.pth'
        encoder2 = Encoder(ENCODER_TYPE, DECODER_TYPE, pretrained=False)

        checkpoint = torch.load(model_path2, map_location=device)
        encoder2.load_state_dict(checkpoint['encoder'])
        encoder2.to(device)

        encoder1.eval()
        encoder2.eval()

        for step, (file_name, images, targets) in tqdm(enumerate(valid_loader)):
            images = images.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                y_preds1 = encoder1(images)
                y_preds2 = encoder2(images)
                y_preds = (y_preds1 + y_preds2)/2

                # prepare mask
                mask = y_preds          
                mask[mask >= 0] = 255
                mask[mask<0] = 0
                mask = mask.cpu()


                for j, m in enumerate(mask):
                    path = os.path.join(target_dir, file_name[j].split('/')[-1].split('.')[0]+'.png')
                    cv2.imwrite(path,np.array(m[0],dtype=np.uint8))

    ref_dir = './DATA/Final_DATA/task02_train/masks'
    file_list = os.listdir(target_dir)
    iou = [0]*len(file_list)
    for i, file_name in enumerate(tqdm(file_list)):
        target_mask = cv2.imread(os.path.join(target_dir,file_name), cv2.IMREAD_GRAYSCALE)
        target_mask[target_mask>0] = 1
        ref_mask = cv2.imread(os.path.join(ref_dir,file_name), cv2.IMREAD_GRAYSCALE)
        ref_mask[ref_mask>0] = 1

        intersection = np.sum(target_mask*ref_mask)
        union = np.sum(target_mask) + np.sum(ref_mask) - intersection
        iou[i] = intersection/union

    sorted_index = list(range(len(iou)))
    sorted_index = sorted(sorted_index, key=lambda x: iou[x])

    polygon_iou = dict()
    for i in sorted_index:
        polygon_iou[file_list[i]] = iou[i]

    with open('./DATA/polygon_iou.json','w') as json_file:
        json.dump(polygon_iou, json_file)

def iou_preprocessing(dataset_path):

    iou_threshold = IOU_THRESHOLD

    polygon_iou = json.load(open('./DATA/polygon_iou.json','rb'))
    iou_dic ={}
    for i, k in enumerate(polygon_iou):
        each ={}
        each['file_id'] = k.split('.')[0]
        each['iou']      = polygon_iou[k]
        iou_dic[i] = each
    iou_pd = pd.DataFrame(iou_dic.values())
    iou_pd = iou_pd[iou_pd.iou>iou_threshold]
    
    train_label = json.load(open(dataset_path + '/labels.json','rb'))
    annotations = train_label['annotations']

    lst = []
    new_dic ={}
    for i,v in enumerate(annotations):
        x,y = [x['x'] for x in v['polygon1']],[x['y'] for x in v['polygon1']]
        each ={}
        pgon = Polygon(zip(x, y))              
        each['counts']    = len(v['polygon1'])
        each['area']      = pgon.area 
        each['file_id'] = v['file_name'].split('.')[0]
        new_dic[i] = each
        lst.append(len(v['polygon1']))
    
    area_pd = pd.DataFrame(new_dic.values())
    area_pd['zscore'] = stats.zscore(area_pd.area)
    area_pd = area_pd[np.abs(area_pd.zscore)<1.2]
    target = pd.merge(iou_pd,area_pd,on='file_id',how='outer').reset_index(drop=True)

    data = pd.DataFrame()
    data['images'] = sorted(['./DATA/Final_DATA/task02_train/images/'+x+'.jpg' for x in target.file_id.values])
    data['masks'] = sorted(['./DATA/Final_DATA/task02_train/masks/'+x+'.png' for x in target.file_id.values])

    FOLDS = 3 # use only 20% data
    kf = KFold(FOLDS, random_state=RANDOM_SEED, shuffle=True)

    data['fold'] = 0
    for fold, (tr_idx, val_idx) in enumerate(kf.split(data)):
        data.loc[val_idx, 'fold'] = fold

    return data

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))