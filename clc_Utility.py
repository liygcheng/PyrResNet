from __future__ import division
import argparse
import time
import os
import types
import math
import pdb
from functools import wraps
from math import exp
import itertools

import numpy as np
#from skimage.measure import structural_similarity as ssim
from skimage.measure import compare_ssim as ssim
from PIL import Image,ImageOps

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from clc_DataLoader import *
import torchvision.models as models
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from tensorboard_logger import configure,log_value
from torch.autograd import Variable



criterion = nn.MSELoss()


def Arguments_Init():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot',help='training data root folder , must containing clean,albedo,shading',default='.')
    parser.add_argument('--batchSize',type = int,help='input batch size',default = 8)
    parser.add_argument('--workers',type = int,help='num of data loading workers',default = 4)
    parser.add_argument('--cuda',action = 'store_true',help='Enable cuda')
    parser.add_argument('--imageSize',type = int , help='the height / width of input image to network',default=256)
    parser.add_argument('--niter',type=int,help='number of epochs of training',default=500)
    parser.add_argument('--gpu_id',type=int,default=0)
    parser.add_argument('--levels',type=int,default=4)
    parser.add_argument('--albedo_model',type=str,default='Models/ResPyramid_Models/albedo/2017-10-27-23-44/')
    parser.add_argument('--shading_model',type=str,default='Models/ResPyramid_Models/shading/2017-10-27-23-44/')

    parser.add_argument('--training_error',type=str,default='Results/LogError/')
    parser.add_argument('--test_error',type=str,default='Results/LogError/')

    opt = parser.parse_args()
    print(opt)
    return opt

def Tensorboard_Init(opt,folder='MPI'):
    #name Models/imageSize_niter_batchSize_ndf
    configure_name  = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))+'_'
    configure_name  = configure_name +  'imageSize_'+str(opt.imageSize)+'_niter_'+str(opt.niter)+'_batchSize_'+      (str(opt.batchSize))
    configure('Results/LossVis/'+folder+'/'+configure_name,flush_secs=5)
    return configure_name


def Data_SceneSplit_Init(opt,status='train'):
    batchSize,shuffle_flag = (opt.batchSize,True)if status=='train' else (1,False)
    transform = clc_Train_Agumentation(size = opt.imageSize) if status=='train' else clc_Test_Agumentation()
    clc_dataset = clc_SceneSplit_DataSet( transform = transform,status=status)

    return torch.utils.data.DataLoader(dataset=clc_dataset,batch_size=batchSize,shuffle=shuffle_flag,num_workers=opt.workers,pin_memory=True)


############################################################

def DumpModel_Init(component):
    if component not in ['albedo','shading']:
        raise NameError("dump model must be one of 'albedo' or 'shading' ")
    currenttime = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
    model_path = 'Results/Models/ResPyramid_Models/' + component + '/'+currenttime

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return model_path
class clc_log_train(object):
    def __init__(self):
        self.loss = self.a_loss = self.s_loss  = 0.0
        self.count = 0
    def step(self,loss,a_loss,s_loss):
        self.loss += loss; self.a_loss += a_loss; self.s_loss += s_loss
        self.count += 1
    def update(self,epoch):
        log_value('loss',self.loss/self.count,epoch)
        log_value('albedo_loss',self.a_loss/self.count,epoch)
        log_value('shading_loss',self.s_loss/self.count,epoch)
    def get_val_list(self):
        val_lists = [self.loss,self.a_loss,self.s_loss]
        return [x/self.count for x in val_lists]

class clc_log_test(object):
    def __init__(self):
        self.A_siMSE  = self.A_siLMSE  = self.A_DSSIM =  0.0
        self.S_siMSE  = self.S_siLMSE  = self.S_DSSIM =  0.0;

        self.count = 0
    def step(self,A_siMSE,A_siLMSE,A_DSSIM,S_siMSE,S_siLMSE,S_DSSIM,batch_channel):
        self.count += batch_channel
        self.A_siMSE  += A_siMSE;self.A_siLMSE += A_siLMSE;self.A_DSSIM  += A_DSSIM
        self.S_siMSE  += S_siMSE;self.S_siLMSE += S_siLMSE;self.S_DSSIM  += S_DSSIM

    def update(self,epoch):
        log_value('albedo_siMSE',self.A_siMSE/self.count,epoch)
        log_value('shading_siMSE',self.S_siMSE/self.count,epoch)

        log_value('albedo_siLMSE',self.A_siLMSE/self.count,epoch)
        log_value('shading_siLMSE',self.S_siLMSE/self.count,epoch)

        log_value('albedo_DSSIM',self.A_DSSIM/self.count,epoch)
        log_value('shading_DSSIM',self.S_DSSIM/self.count,epoch)

    def get_val_list(self):
        val_lists = [self.A_siMSE,self.S_siMSE,self.A_siLMSE,self.S_siLMSE,self.A_DSSIM,self.S_DSSIM]
        return [x/self.count for x in val_lists]

##########################     Loss   Function   ######################

def clc_Loss_albedo(fake,real):
    lambda_tv = 1e-4

    #prob = (1 - (-3.1416*((fake-real)**2)).exp() )**2

    #loss_data = (prob*( (fake-real)**2 )).mean()
    loss_data = criterion(fake,real) + clc_Loss_data(fake,real) + lambda_tv*clc_tv_norm(fake)
    #loss_data = prob.mean()
    return loss_data#criterion(fake,real)#clc_Loss_data(fake,real) #+ lambda_tv*clc_tv_norm(fake)


def clc_Loss_shading(fake,real):

    lambda_tv = 1e-4

    loss_data =  criterion(fake,real) + clc_Loss_data(fake,real) + lambda_tv*clc_tv_norm(fake)
    return loss_data

def clc_Loss_data(fake,real):

#    H, W = fake.size()[2], fake.size()[3]

#    padding = 2
#    tmp_pad = nn.ConstantPad2d((padding, padding, padding, padding))
#    tmp_inversepad = nn.ConstantPad2d((-padding, -padding, -padding, -padding))

#    disp = range(-padding,padding+1)
#    disp  = [x for x in itertools.product(disp,disp)]

#    for h, w in disp:
#        real_tmp = real[:,:,]
#
#    return loss.mean()
    weights,neighbors = get_shift_weight(real)
    space_weight = [0.0838,0.0838,0.0838,0.0838,0.0113,0.0113,0.0113,0.0113,0.6193]

    tmp_weights = []
    for i in range(len(space_weight)):
        tmp_weights.append(weights[i]*space_weight[i])

    tmp_sum = sum(tmp_weights)
    for i,x in enumerate(tmp_weights):
        tmp_weights[i] = x/tmp_sum

    loss_up     = tmp_weights[0]   * ( (fake-neighbors[0])**2  )
    loss_down   = tmp_weights[1]   * ( (fake-neighbors[1])**2  )
    loss_le     = tmp_weights[2]   * ( (fake-neighbors[2])**2  )
    loss_ri     = tmp_weights[3]   * ( (fake-neighbors[3])**2  )
    loss_ul     = tmp_weights[4]   * ( (fake-neighbors[4])**2  )
    loss_ur     = tmp_weights[5]   * ( (fake-neighbors[5])**2  )
    loss_dl     = tmp_weights[6]   * ( (fake-neighbors[6])**2  )
    loss_dr     = tmp_weights[7]   * ( (fake-neighbors[7])**2  )

    loss_center = tmp_weights[8]* ((fake - real)**2)
    loss = loss_up + loss_down + loss_le + loss_ri + loss_center + loss_ul + loss_ur + loss_dl + loss_dr

    return loss.mean()


def get_shift_index(num,shift_value):# shift_value = -1 e.g.
    index = [min(max(0,x+shift_value),num-1) for x in range(num)]
    return Variable(torch.LongTensor(index).cuda())

def get_shift_weight(real):
    sz = real.size()
    ind_up   =  get_shift_index(sz[2],-1)
    ind_down =  get_shift_index(sz[2],1)
    ind_le   =  get_shift_index(sz[3],-1)
    ind_ri   =  get_shift_index(sz[3],1)

    real_up   = (torch.index_select(real,2,ind_up))
    real_down = (torch.index_select(real,2,ind_down))
    real_le   = (torch.index_select(real,3,ind_le))
    real_ri   = (torch.index_select(real,3,ind_ri))
    real_ul   = (torch.index_select(real_up,3,ind_le))
    real_ur   = (torch.index_select(real_up,3,ind_ri))
    real_dl   = (torch.index_select(real_down,3,ind_le))
    real_dr   = (torch.index_select(real_down,3,ind_ri))

    sigma_color = (((real**2).sum())*real.nelement() - (real.sum())**2).div(real.nelement() ** 2)  # adaptive BF

    gauss_sigma_color = -0.5/(sigma_color*sigma_color)

    gauss_sigma_color = gauss_sigma_color.repeat(real.size())

    w_up   =  (((real - real_up)**2)*(gauss_sigma_color)).exp()
    w_down =  (((real - real_down)**2)*(gauss_sigma_color)).exp()
    w_le   =  (((real - real_le)**2)*(gauss_sigma_color)).exp()
    w_ri   =  (((real - real_ri)**2)*(gauss_sigma_color)).exp()
    w_ul   =  (((real - real_ul)**2)*(gauss_sigma_color)).exp()
    w_ur   =  (((real - real_ur)**2)*(gauss_sigma_color)).exp()
    w_dl   =  (((real - real_dl)**2)*(gauss_sigma_color)).exp()
    w_dr   =  (((real - real_dr)**2)*(gauss_sigma_color)).exp()
    w_center = (((real - real)**2)*(gauss_sigma_color)).exp()



    weights   = [w_up,w_down,w_le,w_ri,w_ul,w_ur,w_dl,w_dr,w_center]
    tmp_sum = sum(weights)


    for i,x in enumerate(weights):
        weights[i] = x/tmp_sum

    neighbors = [real_up,real_down,real_le,real_ri,real_ul,real_ur,real_dl,real_dr]

    return weights,neighbors



def clc_tv_norm(input):
    return torch.mean(torch.abs(input[:,:,:,:-1]-input[:,:,:,1:])) + torch.mean(torch.abs(input[:,:,:-1,:]-input[:,:,1:,:]))



##########################     Evaluation  Meterics   ######################
def clc_efficient_siMSE(fake,real,mask):
    B,C,H,W = fake.size()
    bn_ch = B*C ; hw = H*W
    if isinstance(fake,Variable):
        X = (mask*fake).view(bn_ch,H,W).data
        Y = (mask*real).view(bn_ch,H,W).data
        M =  mask.view(bn_ch,H,W).data
    else:
        X = (mask*fake).view(bn_ch,H,W)
        Y = (mask*real).view(bn_ch,H,W)
        M =  mask.view(bn_ch,H,W)
    mse_error = 0.0
    for bc in range(bn_ch):
        if torch.sum(M[bc,:,:])==0:
            continue
        deno = torch.sum(X[bc,:,:]**2)
        nume = torch.sum(Y[bc,:,:]*X[bc,:,:])
        if deno>1e-5:
            alpha = nume/deno
        else:
            alpha = 0
        mse_error += torch.mean((((X[bc,:,:]*alpha) - Y[bc,:,:])**2))
    return mse_error,bn_ch

def clc_efficient_siLMSE(fake,real,mask):

    B,C,H,W = fake.size()
    st = int(W//10)
    half_st = int(st // 2)

    pad_h,pad_w = clc_pad(H,W,half_st)

    tmp_pad     = nn.ZeroPad2d((0,pad_w,0,pad_h))
    tmp_unpad   = nn.ZeroPad2d((0,-pad_w,0,-pad_h))

    X = tmp_pad(fake*mask)
    Y = tmp_pad(mask*real)
    M = tmp_pad(mask)

    idx_jn = (H+pad_h)//half_st
    idx_in = (W+pad_w)//half_st

    LMSE_error = 0
    count = 0

    X_ij = torch.zeros(B,C,half_st*2,half_st*2).cuda()
    Y_ij = torch.zeros(B,C,half_st*2,half_st*2).cuda()
    M_ij = torch.zeros(B,C,half_st*2,half_st*2).cuda()

    for j in range(idx_jn-2):
        for i in range(idx_in-2):
            X_tmp = X[:,:,j*half_st:(j+2)*half_st,i*half_st:(i+2)*half_st]
            Y_tmp = Y[:,:,j*half_st:(j+2)*half_st,i*half_st:(i+2)*half_st]
            M_tmp = M[:,:,j*half_st:(j+2)*half_st,i*half_st:(i+2)*half_st]

            X_ij.copy_(X_tmp.data)

            Y_ij.copy_(Y_tmp.data)
            M_ij.copy_(M_tmp.data)
            batch_error,_ = clc_efficient_siMSE(X_ij,Y_ij,M_ij)
            count += 1
            LMSE_error += batch_error

    return LMSE_error/(count),B*C


def clc_efficient_DSSIM(fake,real,mask):

    B,C,H,W = fake.size()

    bn_ch = B*C

    X = (mask*fake).view(bn_ch,H,W)
    Y = (mask*real).view(bn_ch,H,W)
    M =  mask.view(bn_ch,H,W)

    X = np.transpose(X.data.cpu().numpy(),(1,2,0))
    Y = np.transpose(Y.data.cpu().numpy(),(1,2,0))
    M = np.transpose(M.data.cpu().numpy(),(1,2,0))

    s = 0.0

    for i in range(bn_ch):
        s += (1-ssim(Y[:,:,i],X[:,:,i]))/2

    return s,bn_ch

############   some useful  utilies   ##############

def clc_pad(h,w,st=32):## default st--> 32
    def _f(s):
        n = s//st
        r = s %st
        if r == 0:
            return 0
        else:
            return st-r
    return _f(h),_f(w)

def clc_unpad(X,pad_h,pad_w):
    h = X.size()[2] - pad_h
    w = X.size()[3] - pad_w
    return X[:,:,:h,:w]


if __name__ == "__main__":
    pass


