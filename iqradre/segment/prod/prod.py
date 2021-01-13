
import torch
from ..models.unet import UNet
import pathlib
import PIL
import PIL.Image
from ..data import loader
from . import utils
from PIL import Image


from torchvision.transforms import transforms as VT
from torchvision.transforms import functional as VF


class SegmentationPredictor(object):
    def __init__(self, weight_path=None, device='cpu'):
        self.device = device
        self._load_model()
        
        self.weight = weight_path
        if self.weight != None:
            self._load_state_dict(weight_path)
            
    def _load_state_dict(self, state_dict_path):
        state_dict = torch.load(state_dict_path)
        self.model.load_state_dict(state_dict)
        
    def _load_model(self):
        self.model = UNet(in_chan=1, n_classes=1, start_feat=32)
    
    def _clean_output(self, output): 
        output = output.squeeze()    
        if self.device =="cpu":
            output = output.cpu()
        output = output.detach()
        return output
        
    def predict(self, image, mask_color="#ffffff"):
        if type(image) == str:
            im_path = pathlib.Path(image)
            image = PIL.Image.open(str(im_path))
        
        w, h = image.size
        image_tmft = utils.valid_tmft(image)
        image_tmft = image_tmft.unsqueeze(dim=0)
        output = self.model(image_tmft)
        output = torch.sigmoid(output)
        
        output = self._clean_output(output)
        output = VF.to_pil_image(output)
        mask = VF.resize(output, size=(h,w))

        combined = Image.new("RGBA", (w, h), mask_color)
        combined.paste(image, mask=mask)
        
        return image, mask, combined
        
        