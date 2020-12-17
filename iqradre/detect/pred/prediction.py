import torch
from . import functional as PF
from . import net 
from ..utils import craft_utils
from ..utils import imgproc


import numpy as np

class BoxesPredictor:
    def __init__(self, weight_path, key=None, device=None, dsize=(768, 768), use_refiner=False, weight_refiner=None):
        self.weight_path = weight_path
        self.key = key
        self.dsize = dsize
        self.use_refiner = use_refiner
        self.weight_refiner = weight_refiner
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model: torch.nn.Model = self.load_model()
        if use_refiner:
            self.refine_net = self.load_refine_net()

    def load_model(self):
        return net.load_craft_network(self.weight_path, key=self.key, device=self.device)

    def load_refine_net(self):
        return net.load_refine_net(self.weight_refiner, device=self.device)

    def predict(self, tensor):
        self.model.eval()
        with torch.no_grad():
            y, feature = self.model.forward(tensor)
            y = PF.tensor_minmax_scale(y)  # need tested first

        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        if self.use_refiner:  # not tested yet, result may be broken
            print("refiner not tested yet, it may need to minmax_scale result in tensor before it can be used")
            with torch.no_grad():
                y_refiner = self.refine_net(y, feature)
                score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        return score_text, score_link

    def predict_word_boxes(self, image, text_threshold=0.7, link_threshold=0.3, low_text=0.3, poly=False, mag_ratio=1):
        if type(image) == str:
            image = PF.load_image(image)
            tensor, target_ratio, size_heatmap, img_resized = PF.image_to_tensor(image, mag_ratio=mag_ratio)
            ratio_h = ratio_w = 1 / target_ratio
        elif type(image) == np.ndarray:
#             image = PF.load_image(image)
            tensor, target_ratio, size_heatmap, img_resized = PF.image_to_tensor(image, mag_ratio=mag_ratio)
            ratio_h = ratio_w = 1 / target_ratio
        else:
            raise Exception(f'image must be string of path to file or numpy ndarray type, other type are not supported')

        score_text, score_link = self.predict(tensor)

        # Post-processing
        boxes, polys = craft_utils.get_det_boxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # Coordinate adjustment
        boxes = craft_utils.adjust_result_coordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjust_result_coordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        # Render results (optional)

        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        # boxes = np.sort(boxes, axis=0)
        # boxes = np.sort(boxes, axis=1)
        sorted_bbox = PF.sort_boxes_lrtb(boxes)
        images_patch = PF.boxes_to_images(image, boxes)

        return polys, boxes, images_patch, image, score_text, score_link, ret_score_text

    def predict_char_boxes(self, image_path, low_text=0.3, mag_ratio=1):
        image = PF.load_image(image_path)
        tensor, target_ratio, size_heatmap, img_resized = PF.load_image_tensor(image_path, mag_ratio=mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        score_text, score_link = self.predict(tensor)

        # Post-processing
        boxes = PF.char_bbox(score_text, low_text=low_text)

        # Coordinate adjustment
        boxes = craft_utils.adjust_result_coordinates(boxes, ratio_w, ratio_h)

        images_patch = PF.boxes_to_images(image, boxes)

        return boxes, score_text, score_link, image, images_patch