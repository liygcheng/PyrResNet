"""
Copyright 2017. All Rights Reserved. Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and not-for-profit purposes, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright notice,this paragraph and the following three paragraphs appear in all copies,modifications, and distributions.

Created by Lechao Cheng, Computer Science, Zhejiang University, China. Contact Lechao Cheng(liygcheng@zju.edu.cn) for commercial licensing opportunities.

IN NO EVENT SHALL THE AUTHER(Lechao Cheng) BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF AUTHER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE AUTHER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY,PROVIDED HEREUNDER IS PROVIDED "AS IS". THE AUTHER HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
import math
import numpy as np
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable, Function
import torch.optim as optim
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torchvision.transforms import ToTensor, ToPILImage


def conv1x1(in_channel, out_channel, stride=1, bias=False):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=bias)
def conv3x3(in_channel, out_channel, stride=1, bias=False):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=bias)
def convT3x3(in_channel, out_channel, stride=1, bias=False):
    return nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, output_padding=1, bias=bias)
def conv5x5(in_channel, out_channel, stride=1, bias=False):
    return nn.Conv2d(in_channel, out_channel, kernel_size=5, stride=stride, padding=2, groups=3, bias=bias)
def convT5x5(in_channel, out_channel, stride=1, bias=False):
    return nn.ConvTranspose2d(in_channel, out_channel, kernel_size=5, stride=stride, padding=2, groups=3, output_padding=1, bias=bias)

class clc_ResBlock_NoBN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, ngpu=0):
        super(clc_ResBlock_NoBN, self).__init__()
        self.ngpu = ngpu
        self.conv1 = conv3x3(in_channels, out_channels)
        self.relu1 = nn.ELU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.relu2 = nn.ELU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(residual)
        out += residual
        out = self.relu2(out)
        return out

################### Pyramidal  ResNet    ##############################
class clc_Pyramid(nn.Module):
    def __init__(self, clc_block, num_blocks, nf, ngpu, in_channels=3):
        super(clc_Pyramid, self).__init__()
        #parameters
        self.clc_block = clc_block
        self.ngpu = ngpu
        self.in_channels = in_channels
        main_model = []
        for num in num_blocks:
            main_model.append(self.make_layer(clc_block, nf, num))
        main_model.append(self.make_layer(clc_block, 3, num))
        self.main_model = nn.Sequential(*main_model)
        ## weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

    def make_layer(self, clc_block, out_channels, num_blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels, stride=stride),
                nn.ELU(inplace=True)
            )
        layers = []
        layers.append(clc_block(self.in_channels, out_channels, stride, downsample, ngpu=self.ngpu))
        self.in_channels = out_channels

        for i in range(1, num_blocks):
            layers.append(clc_block(out_channels, out_channels, ngpu=self.ngpu))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.main_model(x)

class clc_Pyramid_Sample(nn.Module):
    def __init__(self, isDownSample=True, level=1, in_channel=3, out_channel=3):
        super(clc_Pyramid_Sample, self).__init__()
        convs = []
        if isDownSample:
            for  k in range(level):
                convs.append(conv5x5(in_channel=3, out_channel=3, stride=2, bias=False))
                convs.append(nn.ELU(inplace=True))
        else:
            convs.append(convT5x5(in_channel=3, out_channel=3, stride=2, bias=False))
            convs.append(nn.ELU(inplace=True))

        self.convs = nn.Sequential(*convs)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
    def forward(self, x):
        return self.convs(x)
###################    Pyramidal  ResNet   ###########################

class clc_Image_Model(object):
    def __init__(self, opt, status='train', lr=0.0005):
        super(clc_Image_Model, self).__init__()
        self.opt = opt
        self.lr = lr
        self.status = status

        self.main_model = nn.ModuleList()
        self.upsample_model = nn.ModuleList()
        self.downsample_model = nn.ModuleList()

        self.main_model_optimizers = []
        self.upsample_optimizers = []
        self.downsample_optimizers = []
        self._init_model()

    def _init_model(self):
        for level in range(self.opt.levels):
            self.main_model.append(clc_Pyramid(clc_block=clc_ResBlock_NoBN, \
                                             num_blocks=[2, 2, 2, 2], nf=1, ngpu=self.opt.gpu_id))
        for level in range(self.opt.levels-1):
            self.downsample_model.append(clc_Pyramid_Sample(isDownSample=True, level=level+1))
            self.upsample_model.append(clc_Pyramid_Sample(isDownSample=False))

        for net in self.main_model:
            self.main_model_optimizers.append(optim.Adam(net.parameters(), lr=self.lr))

        for net in self.upsample_model:
            self.upsample_optimizers.append(optim.Adam(net.parameters(), lr=self.lr))

        for net in self.downsample_model:
            self.downsample_optimizers.append(optim.Adam(net.parameters(), lr=self.lr))

        if self.opt.cuda:
            for i in range(self.opt.levels):
                self.main_model[i] = self.main_model[i].cuda()
            for i in range(self.opt.levels-1):
                self.downsample_model[i] = self.downsample_model[i].cuda()
                self.upsample_model[i] = self.upsample_model[i].cuda()

    def _clamp_weights(self, main_model, Vmin=-0.1, Vmax=0.1):
        for i, x in enumerate(main_model):
            for p in x.parameters():
                p.data.clamp_(Vmin, Vmax)
        return main_model
    def clamp_model_weights(self):
        self.main_model = self._clamp_weights(main_model=self.main_model)
        self.downsample_model = self._clamp_weights(main_model=self.downsample_model)
        self.upsample_model = self._clamp_weights(main_model=self.upsample_model)

    def forward(self, input, mask): #
        fake = self.main_model[self.opt.levels - 1](self.downsample_model[self.opt.levels - 2](input))
        for d_l in range(self.opt.levels-2, -1, -1):
            tmp = input if d_l == 0 else self.downsample_model[d_l-1](input)
            fake = self.main_model[d_l](tmp) + self.upsample_model[d_l](fake)
        return fake

    def optimizer_zerograd(self):
        for optimizer_group in (self.main_model_optimizers, self.upsample_optimizers, self.downsample_optimizers):
            for optimizer in optimizer_group:
                optimizer.zero_grad()
    def optimizer_step(self):
        for optimizer_group in (self.main_model_optimizers, self.upsample_optimizers, self.downsample_optimizers):
            for optimizer in optimizer_group:
                optimizer.step()

    def set_status(self, flag='train'):
        for model in (self.main_model, self.downsample_model, self.upsample_model):
            for i, net in enumerate(model):
                model[i] = net.train() if flag == 'train' else net.eval()

    def optimizer_update_lr(self, lr):
        for optimizer_group in (self.main_model_optimizers, self.upsample_optimizers, self.downsample_optimizers):
            for optimizer in optimizer_group:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

###################     Feature  Encoder   ########################
class VGG_Encoder(nn.Module):
    def __init__(self, gpu_id=0):
        super(VGG_Encoder, self).__init__()
        self.vgg = models.vgg11().features
        self.vgg = self.vgg.cuda(device=gpu_id)
        #self.vgg_cfg = [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M']
        state_dict = torch.load('Models/Vgg_Models/vgg11-bbd30ac9.pth')
        self.index = [0, 3, 6, 8, 11, 13, 16, 18]

        for index in self.index:
            self.vgg[index].weight.data.copy_(state_dict['features.'+str(index)+'.weight'])
            self.vgg[index].bias.data.copy_(state_dict['features.'+str(index)+'.bias'])

        for param in self.vgg.parameters():
            param.requires_grad = False
    def forward(self, fake, real):
        content_loss = 0
        for idx, sub_module in enumerate(self.vgg):
            fake = sub_module(fake)
            real = sub_module(real)
            if idx in self.index:
                content_loss += ((fake-real)**2).mean()
        return content_loss/len(self.index)

##############################    Finally   Module   ####################

############################
if __name__ == "__main__":
    pass
