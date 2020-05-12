""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
import numpy as np

from .unet_parts import *

NUM_IMAGE_PER_SAMPLE = 6

def magic_combine(x, dim_begin, dim_end):
    combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
    return x.view(combined_shape)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.comc = CombConv(6, n_classes)

    def forward(self, x):
        # output_list = []
        # # print(output.shape)
        # for i in np.arange(NUM_IMAGE_PER_SAMPLE):
        #     x1 = self.inc(x[:,i,...])
        #     x2 = self.down1(x1)
        #     x3 = self.down2(x2)
        #     x4 = self.down3(x3)
        #     x5 = self.down4(x4)
        #     x6 = self.up1(x5, x4)
        #     x7 = self.up2(x6, x3)
        #     x8 = self.up3(x7, x2)
        #     x9 = self.up4(x8, x1)
        #     # print(x.shape)
        #     logits = self.outc(x9)
        #     output_list.append(logits)

        # concat = torch.cat((output_list[0],output_list[1],output_list[2],output_list[3],output_list[4],output_list[5]), dim=1)
        # # print(concat.shape)
        # result = self.comc(concat)

        B,N,C,H,W = x.shape
        x0 = x.reshape(B*N,C,H,W)


        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.up1(x5, x4)
        x5 = self.up2(x5, x3)
        x5 = self.up3(x5, x2)
        x5 = self.up4(x5, x1)
        x5 = self.outc(x5)
        B_2,C_2,H_2,W_2 = x5.shape
        x5 = x5.reshape(B_2//N, N, H_2, W_2)


        result = self.comc(x11)

        return result
