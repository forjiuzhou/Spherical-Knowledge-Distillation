dependencies = ['torch', 'torchvision']
from torchvision.models.resnet import resnet18 as _resnet18

def resnet18_skd(pretrained=True, **kwargs):
    """
    Resnet18 model trained with SKD
    """
    checkpoint = 'https://github.com/forjiuzhou/Spherical-Knowledge-Distillation/releases/download/v1/resnet18_skd.pth'
    model = _resnet18()
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False, map_location="cpu", check_hash=True))
    return model
