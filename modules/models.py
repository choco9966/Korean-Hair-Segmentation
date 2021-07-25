import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class Encoder(nn.Module):
    def __init__(self, encoder_name='timm-efficientnet-b3', decoder_name='Unet' , pretrained=False):
        super().__init__()
        if encoder_name in ['se_resnext50_32x4d', 'se_resnext101_32x4d']: 
            encoder_weights = 'imagenet' 
        else: 
            encoder_weights = 'noisy-student' 
            
        if pretrained == False: 
            encoder_weights = None 
        
        if decoder_name == 'Unet': 
            self.encoder = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'UnetPlusPlus':
            self.encoder = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'MAnet': 
            self.encoder = smp.MAnet(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'Linknet': 
            self.encoder = smp.Linknet(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'FPN':
            self.encoder = smp.FPN(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'PSPNet': 
            self.encoder = smp.PSPNet(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'PAN': 
            self.encoder = smp.PAN(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'DeepLabV3': 
            self.encoder = smp.DeepLabV3(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'DeepLabV3Plus': 
            self.encoder = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        else:
            raise ValueError(f"decoder_type : {decoder_name} is not exist")
           
        
    #@autocast()
    def forward(self, x):
        x = self.encoder(x)
        return x


class InferenceEncoder(nn.Module):
    def __init__(self, encoder_name='timm-efficientnet-b3', decoder_name='Unet' , pretrained=False):
        super().__init__()
        if encoder_name in ['se_resnext50_32x4d', 'se_resnext101_32x4d']: 
            encoder_weights = 'imagenet' 
        else: 
            encoder_weights = 'noisy-student' 
        
        if decoder_name == 'Unet': 
            self.encoder = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'UnetPlusPlus':
            self.encoder = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'MAnet': 
            self.encoder = smp.MAnet(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'Linknet': 
            self.encoder = smp.Linknet(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'FPN':
            self.encoder = smp.FPN(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'PSPNet': 
            self.encoder = smp.PSPNet(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'PAN': 
            self.encoder = smp.PAN(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'DeepLabV3': 
            self.encoder = smp.DeepLabV3(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'DeepLabV3Plus': 
            self.encoder = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        else:
            raise ValueError(f"decoder_type : {decoder_name} is not exist")
           
        
    #@autocast()
    def forward(self, x):
        x = self.encoder(x)
        return x
