from torchvision.transforms import transforms as VT
from torchvision.transforms import functional as VF


valid_tmft = VT.Compose([
    VT.Resize((256, 256)),
    VT.Grayscale(),
    VT.ToTensor(),
])