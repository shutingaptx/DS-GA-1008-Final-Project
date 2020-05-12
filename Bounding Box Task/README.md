This is the folder for the bounding box task. There are 3 phases.

### Phase 1: Directly Concatenate Images as Input

Refer to `/YOLOv3`.

#### Image Folder
Move the images of your dataset to `data/custom/images/`.

To get the directly concatenated images, you can refer to `/YOLOv3/image_concat.ipynb`.

#### Annotation Folder
Move your annotations to `data/custom/labels/`. The dataloader expects that the annotation file corresponding to the image `data/custom/images/train.jpg` has the path `data/custom/labels/train.txt`. Each row in the annotation file should define one bounding box, using the syntax `label_idx x_center y_center width height`. The coordinates should be scaled `[0, 1]`, and the `label_idx` should be zero-indexed and correspond to the row number of the class name in `data/custom/classes.names`.

#### Define Train and Validation Sets
In `data/custom/train.txt` and `data/custom/valid.txt`, add paths to images that will be used as train and validation data respectively.

#### Train
To train on the custom dataset run:

```
$ python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data
```

### Phase 2: BEV Images as Input

Refer to `/YOLOv3_BEV`.

The train process is the same as Phase 1.

To get the BEV images, you can refer to `/YOLOv3/IPM_v1.ipynb`.







