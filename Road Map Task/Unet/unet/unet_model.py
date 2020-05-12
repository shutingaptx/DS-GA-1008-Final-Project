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
        #factor = 2 if bilinear else 1
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
        #output_list = []
        # # print(output.shape)
        #for i in np.arange(NUM_IMAGE_PER_SAMPLE):
        #     print(i)
        #     x1 = self.inc(x[:,i,...])
        #     x2 = self.down1(x1)
        #     x3 = self.down2(x2)
        #     x4 = self.down3(x3)
        #     #x5 = self.down4(x4)
        #     #x5 = self.up1(x5, x4)
        #     x4 = self.up2(x4, x3)
        #     x4 = self.up3(x4, x2)
        #     x4 = self.up4(x4, x1)
        #     # print(x.shape)
        #     x4 = self.outc(x4)
        #     output_list.append(x4)

        #concat = torch.cat((output_list[0],output_list[1],output_list[2],output_list[3],output_list[4],output_list[5]), dim=1)
        # # print(concat.shape)
        #return self.comc(concat)

        B,N,C,H,W = x.shape
        x0 = x.reshape(B*N,C,H,W)
        x1 = self.inc(x0)
        del x0
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        del x4        
        x = self.up2(x, x3)
        del x3
        x = self.up3(x, x2)
        del x2
        x = self.up4(x, x1)
        del x1
        x = self.outc(x)
        B_2,C_2,H_2,W_2 = x.shape
        x = x.reshape(B_2//N, N, H_2, W_2)

        return self.comc(x)
