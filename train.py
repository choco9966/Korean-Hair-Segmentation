import os
import time

from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
from modules.utils import *
from modules.preprocessing import *
from modules.solver import train_loop, train_pre_loop

import warnings 
warnings.filterwarnings('ignore')

CONFIG_PATH = './config/config.yaml'
config = load_yaml(CONFIG_PATH)

# TRAIN
TRAIN = config['TRAIN']['train']
N_FOLD = config['TRAIN']['n_fold']
TRN_FOLD = config['TRAIN']['trn_fold']

dataset_path = './DATA/Final_DATA/task02_train'
test_dataset_path = './DATA/Final_DATA/task02_test'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(LOGGER):
    if TRAIN:
        data = iou_preprocessing(dataset_path)
        for fold in range(N_FOLD):
           if fold in TRN_FOLD:
               train_loop(data, fold, LOGGER)


if __name__ == '__main__':
    seed_torch()
    LOGGER = init_logger()
    main(LOGGER)
