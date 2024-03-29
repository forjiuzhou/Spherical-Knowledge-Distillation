# Spherical-Knowledge-Distillation

The code for implementing the SKD https://arxiv.org/abs/2010.07485

## Highlight
1. Simple to implement and fast to train. SKD adds only two lines of code onto Hinton Distillation.
2. High accuracy. SKD can train a ResNet18 with 73% accuracy. 
3. Eases capacity gap problem. SKD can train a highly performance ResNet18 model (72.7% accuracy) with ResNet152 teacher.
4. Very robust with temperature

This code is implemented with apex mixed precision training and dali. Apex and Dali can boost the training speed significantly. The details can be seen at https://github.com/NVIDIA/apex and https://github.com/NVIDIA/DALI. With both apex and dali, one can train ResNet18 on ImageNet in about 20 hours under 4 1080tis.

## Model Release
To download the 73.01% accuracy ResNet18:
```
from torchvision.models.resnet import resnet18
checkpoint = 'https://github.com/forjiuzhou/Spherical-Knowledge-Distillation/releases/download/v1/resnet18_skd.pth'
model = resnet18()
model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False, map_location="cpu", check_hash=True))
```

## Requirement
pytorch
dali 
apex

## Minimal Codes
The configuration of apex and dali could be very messy. To run SKD, you can simply add two lines of code into a Hinton KD implementation, just after the model forwarding. To be noticed, the Cross Entropy loss has to use the normalized logits as input.

```python
output = F.layer_norm(output, torch.Size((num_classes,)), None, None, 1e-7) * multiplier
output_t = F.layer_norm(output_t, torch.Size((num_classes,)), None, None, 1e-7) * multiplier
``` 

Layer normalization uses variance to normalize logits, so the appropriate multiplier can be computed by teacher's logits with torch.std(output_t, dim=1). In most cases, 'multiplier' can be set between 2 to 3. If you use F.normalize, the appropriate multiplier should be computed by torch.norm(output_t, dim=1).

## Training
```
python main.py -a resnet18 --lr 0.01 --distillation --T=4 --epochs 100 --multiplier 2 --fp16 [imagenet-folder with train and val folders]
```
