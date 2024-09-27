import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt
import random
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

def set_pairs(data_path, mode='sequential'):
    path_fix = data_path.copy()
    path_mov = data_path.copy()

    out_path = []
    if mode=='sequential':
        for idx in range(len(path_mov)):
            if idx+1 < len(path_mov):
                out_path.append([path_mov[idx], path_mov[idx+1]])
            else:
                out_path.append([path_mov[idx], path_mov[0]])
    else:
        for path_i in path_mov:
            for path_j in path_fix:
                if path_i==path_j:
                    continue
                out_path.append([path_i, path_j])
    return out_path

class AutoPETTrainDataset(Dataset):
    def __init__(self, data_path, data_names):
        self.path = data_path
        self.data_names = data_names

    def norm_img(self, img):
        img[img < -300] = -300
        img[img > 300] = 300
        norm = (img - img.min()) / (img.max() - img.min())
        return norm

    def norm_suv(self, img):
        img_max = 15.#np.percentile(img, 95)
        img_min = 0.#np.percentile(img, 5)
        norm = (img - img_min)/(img_max - img_min)
        norm[norm < 0] = 0
        norm[norm > 1] = 1
        return norm

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def remap_lbl(self, lbl):
        grouping_table = [[1,], [2, 3,], [4,], [5,], [6,], [7,], [8,], [9,], [10,], [11,], [12,], [13, 14,],
                          [15,], [16,], [17,], [18,], [19,], [20,], [21,], [22,], [23,], [24, 25, 26,], [27,],
                          [28, 29, 30, 31, 32,], [33, 34,], [35, 36,], [37, 38,], [39,]]
        label_out = np.zeros_like(lbl)

        for idx, item in enumerate(grouping_table):
            for seg_i in item:
                label_out[lbl == seg_i] = idx + 1
        return label_out

    def __getitem__(self, index):
        mov_name = self.data_names[index]
        x = nib.load(self.path + '{}_CTRes.nii.gz'.format(mov_name))
        x = self.norm_img(x.get_fdata())
        x_suv = nib.load(self.path + '{}_SUV.nii.gz'.format(mov_name))
        x_suv = self.norm_suv(x_suv.get_fdata())
        x_seg = nib.load(self.path + '{}_CTRes_segsimple.nii.gz'.format(mov_name))
        x_seg = x_seg.get_fdata()
        x_seg = self.remap_lbl(x_seg)#Affine reg does not have this.
        
        x = x[None, ...]
        x_suv = x_suv[None, ...]
        x_seg = x_seg[None, ...]
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        x_suv = np.ascontiguousarray(x_suv)  # [Bsize,channelsHeight,,Width,Depth]
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        x = torch.from_numpy(x)
        x_seg = torch.from_numpy(x_seg)
        x_suv = torch.from_numpy(x_suv)
        return x, x_suv, x_seg

    def __len__(self):
        return len(self.data_names)

class AutoPETFullSegDataset(Dataset):
    def __init__(self, data_path, data_names):
        self.path = data_path
        self.data_names = data_names

    def norm_img(self, img):
        img[img < -300] = -300
        img[img > 300] = 300
        norm = (img - img.min()) / (img.max() - img.min())
        return norm

    def norm_suv(self, img):
        img_max = 15.#np.percentile(img, 95)
        img_min = 0.#np.percentile(img, 5)
        norm = (img - img_min)/(img_max - img_min)
        #norm[norm < 0] = 0
        #norm[norm > 1] = 1
        return norm

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        mov_name = self.data_names[index]
        x = nib.load(self.path + '{}_CTRes.nii.gz'.format(mov_name))
        x = self.norm_img(x.get_fdata())
        x_suv = nib.load(self.path + '{}_SUV.nii.gz'.format(mov_name))
        x_suv = self.norm_suv(x_suv.get_fdata())
        x_seg = nib.load(self.path + '{}_CTRes_seg.nii.gz'.format(mov_name))
        x_seg = x_seg.get_fdata()
        if index == 0:
            print('number of class: {}'.format(len(np.unique(x_seg))))
        x = x[None, ...]
        x_suv = x_suv[None, ...]
        x_seg = x_seg[None, ...]
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        x_suv = np.ascontiguousarray(x_suv)  # [Bsize,channelsHeight,,Width,Depth]
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        x = torch.from_numpy(x)
        x_seg = torch.from_numpy(x_seg)
        x_suv = torch.from_numpy(x_suv)
        return x, x_suv, x_seg

    def __len__(self):
        return len(self.data_names)
    
class AutoPETMMDataset(Dataset):
    def __init__(self, data_path, data_names, is_train=True):
        self.is_train = is_train
        self.path = data_path
        self.data_names = self.data_sort(data_names)
    
    def data_sort(self, data_names):
        if self.is_train:
            return data_names
        else:
            mov_names = data_names[0:-1]
            fix_names = data_names[1:]
            return (mov_names, fix_names)

    def norm_img(self, img):
        img[img < -300] = -300
        img[img > 300] = 300
        norm = (img - img.min()) / (img.max() - img.min())
        return norm

    def norm_suv(self, img):
        img_max = 15.#np.percentile(img, 95)
        img_min = 0.#np.percentile(img, 5)
        norm = (img - img_min)/(img_max - img_min)
        norm[norm < 0] = 0
        norm[norm > 1] = 1
        return norm

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def remap_lbl(self, lbl):
        grouping_table = [[1,], [2, 3,], [4,], [5,], [6,], [7,], [8,], [9,], [10,], [11,], [12,], [13, 14,],
                          [15,], [16,], [17,], [18,], [19,], [20,], [21,], [22,], [23,], [24, 25, 26,], [27,],
                          [28, 29, 30, 31, 32,], [33, 34,], [35, 36,], [37, 38,], [39,]]
        label_out = np.zeros_like(lbl)

        for idx, item in enumerate(grouping_table):
            for seg_i in item:
                label_out[lbl == seg_i] = idx + 1
        return label_out

    def init_data(self, x):
        x = x[None, ...]
        x = np.ascontiguousarray(x)  # [Bsize,channels,Height,,Width,Depth]
        x = torch.from_numpy(x)
        return x
    
    def __getitem__(self, index):
        if self.is_train:      
            mov_name = self.data_names[index]
            fix_list = self.data_names.copy()
            fix_list.remove(mov_name)
            random.shuffle(fix_list)
            fix_name = fix_list[0]
        else:
            mov_names, fix_names = self.data_names
            mov_name = mov_names[index]
            fix_name = fix_names[index]
            
        x = nib.load(self.path + '{}_CTRes.nii.gz'.format(mov_name))
        x = self.norm_img(x.get_fdata())
        x_suv = nib.load(self.path + '{}_SUV.nii.gz'.format(mov_name))
        x_suv = self.norm_suv(x_suv.get_fdata())
        x_seg = nib.load(self.path + '{}_CTRes_segsimple.nii.gz'.format(mov_name))
        x_seg = self.remap_lbl(x_seg.get_fdata())
        
        y = nib.load(self.path + '{}_CTRes.nii.gz'.format(fix_name))
        y = self.norm_img(y.get_fdata())
        y_suv = nib.load(self.path + '{}_SUV.nii.gz'.format(fix_name))
        y_suv = self.norm_suv(y_suv.get_fdata())
        y_seg = nib.load(self.path + '{}_CTRes_segsimple.nii.gz'.format(fix_name))
        y_seg = self.remap_lbl(y_seg.get_fdata())

        x = self.init_data(x)
        x_suv = self.init_data(x_suv)
        x_seg = self.init_data(x_seg)
        
        y = self.init_data(y)
        y_suv = self.init_data(y_suv)
        y_seg = self.init_data(y_seg)
        return x, x_suv, x_seg, y, y_suv, y_seg
    
    def __len__(self):
        if self.is_train:
            return len(self.data_names)
        else:
            return len(self.data_names[0])