#system
from __future__ import print_function

import random
import  os
import numpy as np
from scipy import ndimage as ndi
from clc_Utility import *
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd  import Variable
import torchvision.transforms as transforms

# for visualizing loss
from tensorboard_logger import configure,log_value

#defined by myself

from clc_Model import *


#for debug
import pdb
#pdb.set_trace()


#some initialization implementation in clc_Utility.py
opt = Arguments_Init()# get args from user
configure_name = Tensorboard_Init(opt,'MPI') # log info to visualization
opt.manualSeed = random.randint(1,10000) # fix seed
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

dataloader = Data_SceneSplit_Init(opt)
valloader = Data_SceneSplit_Init(opt,status='test')

albedo_model_path   = DumpModel_Init('albedo')
shading_model_path  = DumpModel_Init('shading')
#some initialization implementation in clc_Utility.py

torch.cuda.set_device(opt.gpu_id)

nc = 3;lr=0.0005
#define network
albedo_model =  clc_Image_Model(opt)
shading_model=  clc_Image_Model(opt)

# levels--> default=4
vgg_loss = VGG_Encoder(gpu_id = opt.gpu_id)


###########################  configure  #################################
criterion = nn.MSELoss()
batch_len = len(dataloader)
training_error = test_error = []
h , w = 436,1024
pad_h,pad_w = clc_pad(h,w,32)
tmp_pad = nn.ReflectionPad2d((0,pad_w,0,pad_h));tmp_inversepad = nn.ReflectionPad2d((0,-pad_w,0,-pad_h))
###########################  configure  #################################



def calc_siError(fake_in,real_in,mask_in):

    if mask_in is None:
        mask_in = (Variable(torch.ones(fake_in.size()))).cuda()
    tmp_mse,batch_ch  = clc_efficient_siMSE(fake_in,real_in,mask_in)
    tmp_lmse,batch_ch  = clc_efficient_siLMSE(fake_in,real_in,mask_in)
    tmp_dssim,batch_ch = clc_efficient_DSSIM(fake_in,real_in,mask_in)
    return tmp_mse,tmp_lmse,tmp_dssim,batch_ch

#############################  utilities #################################

#####################  train pyramidal level ####################
for epoch in range(opt.niter):

    train_info = clc_log_train()
    test_info  = clc_log_test()

    albedo_model.set_status(flag='train')
    shading_model.set_status(flag='train')

    for j,(input,albedo_g,shading_g,mask) in enumerate(dataloader,0): # batchSize * nc * isize * isize

        input_g = Variable(input.clone(),requires_grad=False)
        albedo_g = Variable(albedo_g.clone(),requires_grad=False)
        shading_g = Variable(shading_g.clone(),requires_grad=False)
        mask_g = Variable(mask.clone(),requires_grad=False)

        if opt.cuda:
            input_g,albedo_g,shading_g,mask_g= input_g.cuda(),albedo_g.cuda(),shading_g.cuda(),mask_g.cuda()

        albedo_model.clamp_model_weights();
        shading_model.clamp_model_weights();

        #albedo_optimizer.zero_grad();shading_optimizer.zero_grad();
        albedo_model.optimizer_zerograd()
        shading_model.optimizer_zerograd()

        albedo_fake  = albedo_model.forward(input_g,mask_g);
        shading_fake = shading_model.forward(input_g,None)

        loss_albedo_pyr  = 0.5*vgg_loss(albedo_fake,albedo_g)  + clc_Loss_albedo(albedo_fake,albedo_g)
        loss_shading_pyr = 0.5*vgg_loss(shading_fake,shading_g) + clc_Loss_shading(shading_fake,shading_g)


        input_fake  =   albedo_fake * shading_fake

        loss_common =   criterion(input_fake,input_g)

        ### compute loss
        loss = loss_shading_pyr + loss_albedo_pyr + loss_common
        loss.backward()
        #albedo_optimizer.step(); shading_optimizer.step()
        albedo_model.optimizer_step()
        shading_model.optimizer_step()

        print("Epoch [%d/%d] , Iter [%d/%d] Loss: %.5f = %.5f +  %.5f + %.5f "\
                              %(epoch+1,opt.niter,j+1,batch_len,loss.item(),loss_albedo_pyr.item(),\
                                loss_shading_pyr.item(),loss_common.item()))
        train_info.step(loss=loss.item(),a_loss=loss_albedo_pyr.item(),s_loss=loss_shading_pyr.item())

    if (epoch+1)%100 ==0:
        lr = lr*0.75;
        albedo_model.optimizer_update_lr(lr=lr)
        shading_model.optimizer_update_lr(lr=lr)


    if (epoch+1)%10 == 0:
        torch.save(albedo_model,albedo_model_path+'/epoch_'+str(epoch+1))
        torch.save(shading_model,shading_model_path+'/epoch_'+str(epoch+1))

    train_info.update(epoch)
    training_error.append(train_info.get_val_list())

    if (epoch+1)%10 == 0:
        with open(opt.training_error+configure_name+'_training_error.csv','w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(training_error)

        albedo_model.set_status(flag='eval')
        shading_model.set_status(flag='eval')
        for k,(input,albedo,shading,mask) in enumerate(valloader,0):
            input_g   = Variable(input.clone(),volatile=True)
            albedo_g  = Variable(albedo.clone(),volatile=True)
            shading_g = Variable(shading.clone(),volatile=True)
            mask_g    = Variable(mask.clone(),volatile=True)

            if opt.cuda:
                input_g,albedo_g,shading_g,mask_g = input_g.cuda(),albedo_g.cuda(),shading_g.cuda(),mask_g.cuda()

            input_g,albedo_g,shading_g,mask_g = tmp_pad(input_g),tmp_pad(albedo_g),tmp_pad(shading_g),tmp_pad(mask_g);

            albedo_fake  = albedo_model.forward(input_g,None);
            shading_fake = shading_model.forward(input_g,None)

            input_g,albedo_g,shading_g,mask_g =\
            tmp_inversepad(input_g),tmp_inversepad(albedo_g),tmp_inversepad(shading_g),tmp_inversepad(mask_g);
            albedo_fake,shading_fake  = tmp_inversepad(albedo_fake.clamp(0,1)),tmp_inversepad(shading_fake.clamp(0,1))

            albedo_fake  = albedo_fake*mask_g
            shading_fake = shading_fake

            A_siMSE,A_siLMSE,A_DSSIM, batch_channel1 = calc_siError(albedo_fake,albedo_g,mask_g)
            S_siMSE,S_siLMSE,S_DSSIM, batch_channel2 = calc_siError(shading_fake,shading_g,None)

            test_info.step(A_siMSE,A_siLMSE,A_DSSIM,S_siMSE,S_siLMSE,S_DSSIM,batch_channel1)
            print('k--> ',k,': ',A_siMSE/batch_channel1,A_siLMSE/batch_channel1,\
                  A_DSSIM/batch_channel1,S_siMSE/batch_channel2,S_siLMSE/batch_channel2,S_DSSIM/batch_channel2)
        test_info.update(epoch)
        test_error.append(test_info.get_val_list())
        with open(opt.test_error+configure_name+'_test_error.csv','w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(test_error)

    #####################  train subsequent level ####################
