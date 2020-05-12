import os
import numpy as np
import cv2
import kornia

from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class _BEV():

    def __init__( self, img, view, points_src, points_dst ):
        self.image =  img
        self.view = view
        self.points_src = points_src
        self.points_dst = points_dst
        self.src_img =  self._read()
        self.dst_img = self._convert()

    def _read( self ):
        return (self.image.transpose(1,2,0)*255).astype(int)

    def _convert( self ):

        src_img = kornia.image_to_tensor( self.src_img, keepdim = False )

        dst_h, dst_w = 800, 800

        # Compute perspective transform
        M = kornia.get_perspective_transform(self.points_src, self.points_dst)


        # Image to BEV transformation
        dst_img = kornia.warp_perspective(
            src_img.float(), M, dsize=(dst_h, dst_w), flags='bilinear', border_mode='zeros')

        # remove unwanted portion of BEV image. e.g for FRONT view dst point should not be higher than 450.
        if self.view == 1:
            dst_img[:, :, 400:, :] = 0

        if self.view == 4:
            dst_img[:, :, :400, :] = 0

        if self.view in [0,3]:
            dst_img[:, :, :, 400:] = 0

        if self.view in [2,5]:
            dst_img[:, :, :, :400] = 0

        dst_img = kornia.tensor_to_image(dst_img.byte())
        return dst_img #800*800*3

# CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT
ipm_params = {
    # calibrated using SCENE 111 SAMPLE 11 FRONT image
    1: ( torch.tensor([[ [100.,180.], [123.,163.], [267., 170.], [207., 143.],]]),
                    torch.tensor([[[389, 323], [389, 284], [442, 305.], [461,74.],]])),


    # calibrated using SCENE 124 SAMPLE 0 FRONT_RIGHT image
    2: ( torch.tensor([[[82., 191.], [125., 174.], [257,219.], [264.,190.],]]),
                   torch.tensor([[[438., 342.], [455., 342.], [440, 389], [455, 390],]]) ),

    # calibrated using SCENE 124 SAMPLE 0 BACK_RIGHT image
    5: ( torch.tensor([[[31., 191.], [34., 221.], [188,169.], [229.,183.],]]),
                   torch.tensor([[[456., 405.], [440., 405.], [460, 460], [441, 460],]]) ),

    # calibrated using SCENE 126 SAMPLE 70 FRONT_LEFT image
    0: ( torch.tensor([[[228., 191.], [52., 219.], [202, 174.], [44.,192.],]]),
                   torch.tensor([[[368., 345.], [367., 420.], [348, 345], [347, 420],]]) ),

    # calibrated using SCENE 110 SAMPLE 6 BACK_LEFT image
    3: ( torch.tensor([[[273., 208.], [274., 190.], [48.,170], [94.,166.],]]),
                   torch.tensor([[[358., 400.], [345., 400.], [358, 511],[345, 511], ]]) ),


    # calibrated using SCENE 111 SAMPLE 28 BACK image
    4: ( torch.tensor([[[32., 153.], [192., 167.], [179.,153.], [93,135],]]),
                   torch.tensor([[[464.,540], [391., 484.], [391.,525],[490,769]]]) ),

}

priority_queue = [1,4,0,2,3,5]

def single_combine(images):
    # back left

    back_left0 = images[3].transpose(1,2,0)
    back_left1 = np.flipud(back_left0)
    back_left2 = np.fliplr(back_left1)
    back_left = back_left2.transpose(2,0,1)
    images[3] = back_left

    #back
    back0 = images[4].transpose(1,2,0)
    back1 = np.flipud(back0)
    back2 = np.fliplr(back1)
    back = back2.transpose(2,0,1)
    images[4] = back

    # back right
    back_right0 = images[5].transpose(1,2,0)
    back_right1 = np.flipud(back_right0)
    back_right2 = np.fliplr(back_right1)
    back_right = back_right2.transpose(2,0,1)
    images[5] = back_right

    return torch.as_tensor(images)

def images_prep(sample):
    samples_tensor = torch.stack(sample)
    samples = samples_tensor.numpy()
    output = torch.zeros(samples.shape[0],3,800,800)

    for i in range(samples.shape[0]):
        currrent_sample = samples[i]
        image_pre = single_combine(currrent_sample)
        image_1 = torchvision.utils.make_grid(image_pre, nrow=3, padding=0).numpy()
        image_2 = image_1.swapaxes(-2,-1)[...,::-1,:].transpose(1,2,0)
        image_3 = cv2.resize(image_2, dsize=(800, 800), interpolation=cv2.INTER_CUBIC).transpose(2,0,1)
        output[i] = torch.as_tensor(image_3)

    return output

def masks_prep(road_image):
    return torch.as_tensor(np.expand_dims(torch.stack(road_image), axis=1))

def BEV(sample):
    samples_tensor = torch.stack(sample)
    samples = samples_tensor.numpy()
    output = torch.zeros(samples.shape[0],3,800,800)

    for i in range(samples.shape[0]):
        currrent_sample = samples[i]

        bevs = {}
        for view in ipm_params:

            points_src, points_dst = ipm_params[view]
            bevs[view] = _BEV(currrent_sample[view], view, points_src, points_dst )
#             plt.imshow(currrent_sample[view].transpose(1,2,0))

        out = None
        for view in priority_queue:
            if view in bevs:
                if out is None:
                    out = bevs[view].dst_img.copy()
                else:
                    new_layer = bevs[view].dst_img
                    mask = (out.sum(2) == 0).astype(np.uint8)
                    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                    out += new_layer * mask

        out_rotate = out.transpose(2,0,1).swapaxes(-2,-1)[...,::-1]
        output[i] = torch.as_tensor(out_rotate.copy())

    return output
