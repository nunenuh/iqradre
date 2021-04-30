from numpy.lib.arraypad import pad
import pandas as pd
import numpy as np


from iqradre.detect.pred import BoxesPredictor

from iqradre.recog.prod import TextPredictor, TesseractPredictor
from iqradre.recog.prod import utils as text_utils

import transformers
from transformers import BertTokenizer
from iqradre.extract.prod.prod import Extractor



import matplotlib.pyplot as plt
# from iqradre.detect.ops import boxes as boxes_ops
# from iqradre.detect.ops import box_ops
from iqradre.extract.ops import boxes_ops

from iqradre.segment.prod import SegmentationPredictor

from deskew import determine_skew
from skimage.transform import rotate
import imutils
import cv2 as cv
from . import utils
import torch


torch.cudnn.enabled = True
torch.cudnn.benchmark = True

class IDCardPredictor(object):
    def __init__(self, config, device='cpu', use_tesseract=False):
        self.config = config
        self.device = device
        self.use_tesseract = use_tesseract
        
        self._init_model()
    
    def _init_model(self):
        print(f'INFO: Load all model, please wait...')
        self.segmentor = None
        if self.config.get('segmentor', False):
            self.segmentor = SegmentationPredictor(weight_path=self.config['segmentor'], device=self.device)
        self.boxes_detector = BoxesPredictor(weight_path=self.config['detector'], device=self.device)
        
        if self.use_tesseract:
            self.text_recognitor = TesseractPredictor()
        else:
            self.text_recognitor = TextPredictor(weight_path=self.config['recognitor'], device=self.device)
        
        self.tokenizer = BertTokenizer.from_pretrained(self.config["tokenizer"])
        self.info_extractor = Extractor(tokenizer=self.tokenizer, weight=self.config['extractor'], device='cpu')
        print(f'INFO: All model has been loaded!')
        
    def _clean_boxes_result(self, boxes_result, min_size_percent=2):
        polys, boxes, images_patch, img, score_text, score_link, ret_score_text = boxes_result
        
        #find max
        sizes = [im.shape[1] for im in images_patch]
        max_index, max_value = max(enumerate(sizes), key=lambda x: x[1])

        #create percent by max size
        percent_size = [int(size/max_value * 100) for size in sizes]

        #exclude with minimum_size
        remove_indices = [i for i, psize in enumerate(percent_size) if psize<=min_size_percent]
        # images_patch = np.delete(images_patch, remove_indices).tolist()
        new_boxes = []
        new_images_patch = []
        for i in range(len(images_patch)):
            if not i in remove_indices:
                new_boxes.append(boxes[i])
                new_images_patch.append(images_patch[i])
        new_boxes = np.array(new_boxes)
        boxes_result = polys, new_boxes, new_images_patch, img, score_text, score_link, ret_score_text
        
        return boxes_result
    
    def _resize_normalize(self, image:np.ndarray, dsize=(750, 1000), pad_color=0):
        try:
            outimg = utils.resize_pad(image, size=dsize, pad_color=pad_color)
        except:
            h,w = image.shape[:2]
            print(f'resize exception ori size:({h},{w})')
            ratio = h/w
            if ratio<1.3:
                nh = int(h * 1.3)
                dim = (nh, w)
                outimg = cv.resize(image, dim, interpolation=cv.INTER_LINEAR)
                size = outimg.shape[:2]
                print(f'resize Exception new size: {size}')
        
        return outimg
    
    def predict(self, impath, resize=True, dsize=(1500,2000),
                text_threshold=0.7, link_threshold=0.3, low_text=0.5, 
                min_size_percent=5,):
        segment_img = self._segment_predictor(impath)
        rot_img, angle = self._auto_deskew(segment_img, resize=resize)
        normalized_img = self._resize_normalize(rot_img, dsize=dsize)
        boxes_result = self._detect_boxes(normalized_img, 
                                          text_threshold=text_threshold,
                                          link_threshold=link_threshold,
                                          low_text=low_text)
        boxes_result = self._clean_boxes_result(boxes_result, min_size_percent=min_size_percent)
        polys, boxes, images_patch, img, score_text, score_link, ret_score_text = boxes_result
        
        boxes_list = boxes_ops.to_xyminmax_batch(boxes, to_int=True).tolist() 
        
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
            'segment_image': segment_img,
            'rotated_image': rot_img,
            'images_patch': images_patch,
            'boxes': boxes_list,
            'texts': text_list,
            'score_text': score_text,
            'score_list': score_link,
            'score': score_text+score_link,
        }
        
    def _segment_predictor(self, impath):
        if self.config.get('segmentor', False):
            image, mask, combined = self.segmentor.predict_canvas(impath, mask_color="#000000")
            
            import matplotlib.pyplot as plt
            plt.imshow(combined)
            print('segment_predictor size ==>',combined.size)
            
            combined = combined.convert("RGB")
            result = np.array(combined).astype(np.uint8)
            return result
        else:
            return impath
        
    def _detect_boxes(self, impath, text_threshold=0.7, link_threshold=0.3, low_text=0.5, min_size_percent=5):
        result = self.boxes_detector.predict_word_boxes(impath, 
                                                        text_threshold=text_threshold, 
                                                        link_threshold=link_threshold, 
                                                        low_text=low_text)
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
        
        
 