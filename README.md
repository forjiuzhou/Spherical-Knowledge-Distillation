# Spherical-Knowledge-Distillation

The code for implementing the SKD arxiv:2010.07485

## Highlight
1. Simple and fast. SKD adds only two lines of code onto Hinton Distillation.
2. High accuracy. SKD can train a ResNet18 with 73% accuracy. 
3. Eases capacity gap problem. SKD can train a highly performance ResNet18 model (72.7% accuracy) with ResNet152 as teacher.
4. Very robust with temperature

This code is implemented with apex mixed precision training and dali. Apex and Dali can boost the training speed significantly. The details can be seen at https://github.com/NVIDIA/apex and https://github.com/NVIDIA/DALI. With both apex and dali, one can train ResNet18 on ImageNet in about 20 hours under 4 1080tis.

## Requirement
pytorch
dali 
apex

## Minimal Codes
The configuration of apex and dali could be very messy. To run SKD, you can simply add two lines of code into a Hinton KD implementation, just after the model forwarding.

```
output = F.layer_norm(output, torch.Size((num_classes,)), None, None, 1e-7) * multiplier
output_t = F.layer_norm(output_t, torch.Size((num_classes,)), None, None, 1e-7) * multiplier
``` 
'multiplier' can be set between 2 to 3.

## Training

python main.py -a resnet50 --lr 0.01 --distillation --LN --T=4 --epochs 100 --multiplier 2 --fp16 [imagenet-folder with train and val folders]
