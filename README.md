# DS-GA-1008-Final-Project

This is the final project of DS-GA 1008 Deep Learning 2020Spring.

This project demonstrates the implementation of UNet and YOLOv3 frameworks for solving roadmap segmentation and vehicle detection tasks separately. 
Both raw images and concatenated images(direct concatenation and bird-eye view(BEV)) are applied to manage the input styles.
Self-supervised learning by Jigsaw Puzzle was uti-lized to pretrain the encoder Resnet34 and feature extractor Darknet53.

## Reference
* [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
* [PYOLOv3_PyTorch](https://github.com/BobLiu20/YOLOv3_PyTorch)
* [UNet PyTorch](https://github.com/milesial/Pytorch-UNet)
* IPM: https://torchgeometry.readthedocs.io/en/latest/warp_perspective.html
