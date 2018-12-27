"""
Copyright 2017. All Rights Reserved. Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and not-for-profit purposes, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright notice,this paragraph and the following three paragraphs appear in all copies,modifications, and distributions.

Created by Lechao Cheng, Computer Science, Zhejiang University, China. Contact Lechao Cheng(liygcheng@zju.edu.cn) for commercial licensing opportunities.

IN NO EVENT SHALL THE AUTHER(Lechao Cheng) BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF AUTHER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE AUTHER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY,PROVIDED HEREUNDER IS PROVIDED "AS IS". THE AUTHER HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
import glob
import os
#import pdb
import random
import math

import csv
import cv2

import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

# Sintel scene names
sintel_scenes = dict(
    train=['alley_1', 'bamboo_1', 'bandage_1', 'cave_2', 'market_2', 'market_6', 'shaman_2', 'sleeping_1', 'temple_2'],
    test=['alley_2', 'bamboo_2', 'bandage_2', 'cave_4', 'market_5', 'mountain_1', 'shaman_3', 'sleeping_2', 'temple_3'])

###################    Base dataset setting   ##################

class clc_Base_DataSet(data.Dataset):
    ''' base dataset class for loading mpi intrinsic images '''
    def __init__(self, imageroot='../data/DataSet/MPI_v2/sintel/images/', transform=None, status='train'):
        self.imageroot = imageroot
        self.transform = transform
        self.configure = ''
        self.len = 0
        self.status = status
        self.total_lists = []
        self.RS_lists = []
        self.albedo_lists = []
        self.shading_lists = []
        self.albedo_mask_lists = []
        self.image_indexs = {}
    def get_sorted_filenames(self, folder, ext='*.png'):
        '''get images by ext and return sorted filename list'''
        if folder[-1] != '/':
            folder = folder + '/'
        filenames = glob.glob(folder+ext)
        filenames.sort(key=lambda x: (x[:-4]))
        return filenames
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        pass
    def check_valid(self, input_len, albedo_len, shading_len, albedo_mask_len):
        '''check if it is equal for several list length'''
        assert len(input_len) == len(albedo_len) == len(shading_len) == len(albedo_mask_len)
    def clear_lists(self):
        '''clear variables, prevent the trouble from accumulation'''
        self.RS_lists = []
        self.albedo_lists = []
        self.shading_lists = []
        self.albedo_mask_lists = []
    def get_total_filelists(self):
        '''get mpi data path and dump to configure files'''
        self.total_lists = []
        if not os.path.exists(self.configure):
            subclass = sintel_scenes['train'] + sintel_scenes['test']
            with open(self.configure, 'w') as csvfile:
                writer = csv.writer(csvfile)
                for sub_class in subclass:
                    input_tmp = self.get_sorted_filenames(self.imageroot +'RS/'+sub_class)
                    albedo_tmp = self.get_sorted_filenames(self.imageroot +'albedo/'+sub_class)
                    shading_tmp = self.get_sorted_filenames(self.imageroot +'shading/'+sub_class)
                    albedo_mask_tmp = self.get_sorted_filenames(self.imageroot +'albedo_defect_mask/'+sub_class)
                    for (input_name, albedo_name, shading_name, mask_name) in zip(input_tmp, albedo_tmp, shading_tmp, albedo_mask_tmp):
                        writer.writerow([input_name, albedo_name, shading_name, mask_name])
                        self.total_lists.append([input_name, albedo_name, shading_name, mask_name])
        else:
            self.total_lists = []
            with open(self.configure, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for (input_name, albedo_name, shading_name, mask_name) in reader:
                    self.total_lists.append([input_name, albedo_name, shading_name, mask_name])
        self.len = len(self.total_lists)
    def get_status_filenames(self):
        '''get train or test data set by status
           noted:  self.image_indexs  must be assigned before calling this function
        '''
        self.clear_lists()
        for _, image_index  in enumerate(self.image_indexs[self.status]):
            image_index = int(float(image_index))
            self.RS_lists.append(self.total_lists[image_index][0])
            self.albedo_lists.append(self.total_lists[image_index][1])
            self.shading_lists.append(self.total_lists[image_index][2])
            self.albedo_mask_lists.append(self.total_lists[image_index][3])
        self.check_valid(self.RS_lists, self.albedo_lists, self.shading_lists, self.albedo_mask_lists)
        self.len = len(self.RS_lists)

#################################################################################

class clc_SceneSplit_DataSet(clc_Base_DataSet):
    '''MPI scene split'''
    def __init__(self, imageroot='../data/DataSet/MPI_v2/sintel/images/', transform=None, status='train'):
        super(clc_SceneSplit_DataSet, self).__init__(imageroot, transform, status)
        self.status = status
        self.n_repeat = 4
        self.configure = self.imageroot+'SceneSplitFilenames.csv'
        self.permutation = self.imageroot +'SceneSplitPermutation.csv'
        self.get_total_filelists()
        self.get_status_index()
        self.get_status_filenames()
    def get_status_index(self):
        '''get scene status dataset index from self.permutation'''
        if os.path.exists(self.permutation):
            with open(self.permutation, 'r') as csvfile:
                reader = csv.reader(csvfile)
                indexs = [x for x in reader]
                self.image_indexs['full'] = indexs[0]
                self.image_indexs['train'] = indexs[1]
                self.image_indexs['test'] = indexs[2]
        else:
            with open(self.permutation, 'w') as csvfile:
                writer = csv.writer(csvfile)
                split_index = 0
                for sub_class in sintel_scenes['train']:
                    split_index += len(glob.glob(self.imageroot+'RS/'+sub_class+'/*.png'))
                self.image_indexs['full'] = range(self.len)
                self.image_indexs['train'] = (range(self.len)[:split_index])*self.n_repeat
                self.image_indexs['test'] = range(self.len)[split_index:]
                writer.writerow(self.image_indexs['full'])
                writer.writerow(self.image_indexs['train'])
                writer.writerow(self.image_indexs['test'])
    def __getitem__(self, index):
        input_image = Image.open(self.RS_lists[index]).convert('RGB')
        albedo_image = Image.open(self.albedo_lists[index]).convert('RGB')
        shading_image = Image.open(self.shading_lists[index]).convert('RGB')
        albedo_mask = Image.open(self.albedo_mask_lists[index]).convert('RGB')
        return self.transform(input_image, albedo_image, shading_image, albedo_mask)
    def __len__(self):
        return self.len

##########################    data  preprocessing   #########################
class clc_Test_Agumentation(object):
    '''MPI test dataset prepocessing'''
    def __init__(self):
        self.toTensor = transforms.ToTensor()
    def __call__(self, input_image, albedo_image, shading_image, mask):
        return self.toTensor(input_image), self.toTensor(albedo_image), self.toTensor(shading_image), self.toTensor(mask)
class clc_Train_Agumentation(object):
    '''MPI naive training dataset prepocessing'''
    def __init__(self, size=256, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    def __call__(self, input_image, albedo_image, shading_image, mask=None):
        im_w, im_h = input_image.size[0], input_image.size[1]
        # random crop
        area = im_w * im_h
        target_area, aspect_ratio = random.uniform(0.2, 0.8)*area, random.uniform(3./4, 4./3)
        tmp_w, tmp_h = int(round(math.sqrt(target_area * aspect_ratio))), int(round(math.sqrt(target_area / aspect_ratio)))
        tmp_w, tmp_h = (tmp_h, tmp_w) if random.random() < 0.5 else (tmp_w, tmp_h)
        if tmp_w <= im_w and tmp_h <= im_h:
            start_x, start_y = random.randint(0, im_w-tmp_w), random.randint(0, im_h-tmp_h)
            input_image = input_image.crop((start_x, start_y, start_x+tmp_w, start_y+tmp_h))
            albedo_image = albedo_image.crop((start_x, start_y, start_x+tmp_w, start_y+tmp_h))
            shading_image = shading_image.crop((start_x, start_y, start_x+tmp_w, start_y+tmp_h))
        im_w, im_h = input_image.size[0], input_image.size[1]
        # random scale to [0.8 - 1.2]
        scale_to_size = int(random.uniform(0.8, 1.2) * min(im_w, im_h))
        clc_scale = transforms.Scale(scale_to_size, interpolation=self.interpolation)
        input_image, albedo_image, shading_image = clc_scale(input_image), clc_scale(albedo_image), clc_scale(shading_image)
        # random left-right flip  with probability 0.5
        if random.random() < 0.5:
            input_image = input_image.transpose(Image.FLIP_LEFT_RIGHT)
            albedo_image = albedo_image.transpose(Image.FLIP_LEFT_RIGHT)
            shading_image = shading_image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            input_image = input_image.transpose(Image.FLIP_TOP_BOTTOM)
            albedo_image = albedo_image.transpose(Image.FLIP_TOP_BOTTOM)
            shading_image = shading_image.transpose(Image.FLIP_TOP_BOTTOM)

        input_image = input_image.resize((self.size, self.size), self.interpolation)
        albedo_image = albedo_image.resize((self.size, self.size), self.interpolation)
        shading_image = shading_image.resize((self.size, self.size), self.interpolation)

        albedo_image, shading_image = np.array(albedo_image), np.array(shading_image)
        mask = np.repeat((albedo_image.mean(2) != 0).astype(np.uint8)[..., np.newaxis]*255, 3, 2)
        input_image = (albedo_image.astype(np.float32)/255)*(shading_image.astype(np.float32)/255)*255

        albedo_image = (torch.from_numpy(albedo_image.transpose((2, 0, 1)))).float().div(255)
        shading_image = (torch.from_numpy(shading_image.transpose((2, 0, 1)))).float().div(255)
        mask = (torch.from_numpy(mask.transpose((2, 0, 1)))).float().div(255)
        input_image = (torch.from_numpy(input_image.transpose((2, 0, 1)))).float().div(255)
        return input_image, albedo_image, shading_image, mask
if __name__ == "__main__":
    pass

