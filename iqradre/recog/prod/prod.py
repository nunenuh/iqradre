import string
import torch
from iqradre.recog.utils import AttnLabelConverter
import torchvision.transforms as VT
from iqradre.recog import transforms as NT
from iqradre.recog.models import crnn_v1
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader

from PIL import Image
import numpy as np
import pytesseract
from typing import *



class TextPredictor(object):
    def __init__(self, weight_path, device='cpu'):
        self.weight_path = weight_path
        self.device = device
        self._load_config()
        self._load_model()
        
    def _load_config(self):
        self.character = string.printable[:-6]
        self.converter = AttnLabelConverter(self.character)
        self.num_class = len(self.converter.character)
        self.batch_max_length = 25
        self.img_size = (32, 100)
        
        
    def _load_model(self):
        state_dict = torch.load(self.weight_path, map_location=torch.device(self.device))
        self.model = crnn_v1.OCRNet(num_class=self.num_class, im_size=self.img_size, hidden_size=256)
        self.model.load_state_dict(state_dict)
        
    def _predict(self, images:list):
        dloader = self._transform_loader(images)
        images = next(iter(dloader))
        batch_size = images.shape[0]
        
        
        length = torch.IntTensor([self.batch_max_length] * batch_size)
        preds = self.model(images)
        preds = preds[:, :self.batch_max_length, :]
        _, preds_index = preds.max(2)
        preds_str = self.converter.decode(preds_index, length)
        preds_clean = self._clean_prediction(preds_str)    
        
        return preds_clean
    
    def predict(self, images):
        return self._predict(images)
    
    def _clean_prediction(self, preds_str):
        out = []
        for prd_st in preds_str:
            word = prd_st.split("[s]")[0]
            out.append(word)
        return out
    
    def _transform_loader(self, images):
        transform = VT.Compose([
            VT.ToPILImage(),
            VT.Grayscale(),
            NT.ResizeRatioWithRightPad(size=self.img_size),
            VT.ToTensor(),
            VT.Normalize(mean=(0.5, ), std=(0.5, ))
        ])
        
        out = []
        for image in images:
            timg = transform(image)
            out.append(timg)
            
        return DataLoader(out, batch_size=len(out))
    
class TesseractPredictor(object):
    def __init__(self, lang='eng', use_custom_config=True, oem=1, psm=7):
        self.lang = lang
        self.oem = oem
        self.psm = psm
        self.use_custom_config = use_custom_config
        self.init_config()
        
    def init_config(self):
        self.config = f'--oem {self.oem} --psm {self.psm}'
        
    
    def _recognize(self, image: np.ndarray):
        if self.use_custom_config:
            return pytesseract.image_to_string(image, lang=self.lang, config=self.config)
        else:
            return pytesseract.image_to_string(image, lang=self.lang)

    def _predict(self, images):
        outputs = []
        for image in images:
            text = self._recognize(image)
            outputs.append(text)
            
        return outputs
        
    def predict(self, images: List[np.ndarray])->List[str]:
        return self._predict(images)
        
        
        
        
    
        