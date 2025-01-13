import logging
import os
import numpy as np
import os
import prepare_data.degradations as degradations
import torch
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, 
                                        adjust_hue, adjust_saturation, normalize)

def setup_logger(log_file='./logs/data_pre.log',name='data_pre', level=logging.INFO):
    """Function setup as many loggers as you want"""
    if os.path.exists(log_file):
        os.remove(log_file)
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def color_jitter(img, shift):
    """jitter color: randomly jitter the RGB values, in numpy formats"""
    jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
    img = img + jitter_val
    img = np.clip(img, 0, 1)
    return img


def color_jitter_pt(img, brightness, contrast, saturation, hue):

    """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
    fn_idx = torch.randperm(4)
    for fn_id in fn_idx:
        if fn_id == 0 and brightness is not None:
            brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
            img = adjust_brightness(img, brightness_factor)

        if fn_id == 1 and contrast is not None:
            contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
            img = adjust_contrast(img, contrast_factor)

        if fn_id == 2 and saturation is not None:
            saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
            img = adjust_saturation(img, saturation_factor)

        if fn_id == 3 and hue is not None:
            hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
            img = adjust_hue(img, hue_factor)
    return img