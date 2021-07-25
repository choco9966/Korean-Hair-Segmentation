import albumentations as A
from albumentations.pytorch import ToTensorV2

from modules.utils import load_yaml

CONFIG_PATH = './config/config.yaml'
config = load_yaml(CONFIG_PATH)

# TRAIN
AUGMENTATION = config['TRAIN']['augmentation']

def get_transforms_preprocessing(*, data):
    if data == 'train':
        return A.Compose([
                A.HorizontalFlip(p=0.5), 
                A.OneOf([
                    A.RandomContrast(),
                    A.RandomGamma(),
                    A.RandomBrightness(),
                    ], p=0.3),
                A.OneOf([
                    A.GridDistortion(),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                    ], p=0.3),
                A.GridDropout(p=0.1), 
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

def get_transforms_train(*, data):
    if data == 'train':
        if AUGMENTATION == True:
           return A.Compose([
                   A.OneOf([
                       A.HueSaturationValue(15, 25, 0),
                   ],p=0.1),
                   A.OneOf([
                       A.RandomContrast(),
                       A.RandomGamma(),
                       A.RandomBrightness(),
                       ], p=0.1),
                   A.OneOf([
                       A.GridDistortion(),
                       A.OpticalDistortion(),
                       A.GaussNoise(),
                   ], p=0.1),
                   A.HorizontalFlip(p=0.1),
                   A.Cutout(),
                   A.Normalize(
                   mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)
               ),
               ToTensorV2(transpose_mask=True)
           ],p=1.)
        else:
            return A.Compose([
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

def get_transforms_inference(*, data):
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
