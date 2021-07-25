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

# PREPROCESSING
PREPROCESSING = config['PREPROCESSING']['preprocessing']
PRE_N_FOLD = config['PREPROCESSING']['n_fold']
PRE_TRN_FOLD = config['PREPROCESSING']['trn_fold']

dataset_path = './DATA/Final_DATA/task02_train'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(LOGGER):
    if PREPROCESSING:
        make_masks(dataset_path)
        make_npixels_data(dataset_path)
        data_pre = make_pre_data()
        print('data loaded')
        print('Complete preprocessing')

        # train for preprocessing
        for fold in range(PRE_N_FOLD):
            if fold in PRE_TRN_FOLD:
                train_pre_loop(data_pre, fold, LOGGER)
            
        calculate_iou(data_pre, device)
        print('Complete calculate iou')


if __name__ == '__main__':
    seed_torch()
    LOGGER = init_logger()
    main(LOGGER)
