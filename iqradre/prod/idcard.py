import pandas as pd
import numpy as np


from iqradre.detect.pred import BoxesPredictor

from iqradre.recog.prod import TextPredictor
from iqradre.recog.prod import utils as text_utils

import transformers
from transformers import BertTokenizer
from iqradre.extract.prod.prod import Extractor

import matplotlib.pyplot as plt
from iqradre.detect.ops import boxes as boxes_ops
from iqradre.detect.ops import box_ops

from deskew import determine_skew
from skimage.transform import rotate
import imutils



class IDCardPredictor(object):
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        
        self._init_model()
    
    def _init_model(self):
        print(f'INFO: Load all model, please wait...')
        self.boxes_detector = BoxesPredictor(weight_path=self.config['detector'], device=self.device)
        self.text_recognitor = TextPredictor(weight_path=self.config['recognitor'], device=self.device)
        
        self.tokenizer = BertTokenizer.from_pretrained(self.config["tokenizer"])
        self.info_extractor = Extractor(tokenizer=self.tokenizer, weight=self.config['extractor'], device=self.device)
        print(f'INFO: All model has been loaded!')
        
    def predict(self, impath, resize=True):
        rot_img, angle = self._auto_deskew(impath, resize=resize)
        boxes_result = self._detect_boxes(rot_img)
        polys, boxes, images_patch, img, score_text, score_link, ret_score_text = boxes_result
        boxes_list = box_ops.batch_box_coordinate_to_xyminmax(boxes, to_int=True).tolist() 
        
        text_list =  self.text_recognitor.predict(images_patch)
        
        data_annoset = text_utils.build_annoset(text_list, boxes)
        data_annoset = sorted(data_annoset, key = lambda i: (i['bbox'][1], i['bbox'][0]))
        
        data, clean = self.info_extractor.predict(data_annoset)  
        dframe = pd.DataFrame(clean)
        
#         data, dframe, img, boxes_list, text_list, score_text, score_list
        
        return {
            'prediction': data,
            'dataframe':dframe,
            'image': img,
            'images_patch': images_patch,
            'boxes': boxes_list,
            'texts': text_list,
            'score_text': score_text,
            'score_list': score_link,
            'score': score_text+score_link,
        }
        
    def _detect_boxes(self, impath):
        result = self.boxes_detector.predict_word_boxes(impath, text_threshold=0.3, low_text=0.2)
        polys, boxes, images_patch, img, score_text, score_link, ret_score_text = result
        return result
        
    def _auto_deskew(self, impath, resize=False):
        result = self._detect_boxes(impath)
        polys, boxes, images_patch, img, score_text, score_link, ret_score_text = result
        
        angle = determine_skew(score_text+score_link)
        rotated_img = rotate(img, angle, resize=True)
        
        rotated_img = (rotated_img * 255).astype(np.uint8)
        
        if resize:
            shape = rotated_img.shape[:2]
            max_index = shape.index(max(shape))
            if max_index == 1:
                rotated_img = imutils.resize(rotated_img, width=1000)
            else:
                rotated_img = imutils.resize(rotated_img, height=1000)
        
        return rotated_img, angle
        
        
 