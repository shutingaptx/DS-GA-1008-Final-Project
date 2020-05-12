This is the folder for the road map task. There are 2 schemes. Each scheme contains 3 phases.

## Scheme 1: Change Input Image Styles
### Phase 1: Directly Concatenate Images as Input
Refer to `/Unet_DirectConcat` and run 
```bash
python train.py -e 10 -b 2
```
### Phase 2 & 3: BEV Images as Input
Refer to `/Unet_BEV` and run 
```bash
python train.py -e 15 -b 2
```
To get the BEV images, you can refer to `/Unet_BEV/IPM_v1.ipynb`.

## Scheme 2: Change Encoders and Pretrained Encoder
### Phase 1: Original encoder
Refer to `/Unet` and run
 ```bash
 python train.py -e 15 -b 1 -l 0.005
 ```
 
 ### Phase 2 & 3: ResNet34 and Pretrained ResNet34
 Refer to  `/Resnet_Unet`.
 For ResNet34 without pretraining, run
 ```bash
 python train.py -e 15 -b 2 -l 0.005
 ```
 For pretrained ResNet34, run
 ```bash
 python -u train.py -e 20 -b 2 -l 0.001 -f 'jigsaw2_epoch_7.pth'
 ```
 Here, `jigsaw2_epoch_7.pth` saves the state dict of the pretrained model. The file can be get by running following commands in folder `/Jigsaw_roadmap`.
 ```bash
 python JigsawTrain.py  --epochs 10 --data_path '/scratch/ah4734/data/data'
 ```
