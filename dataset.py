import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter


def preprocess_transform(return_dict, 
        preprocess_crop_size, preprocess_crop_location):

    for key in ['img', 'pos_anchor', 'neg_anchor', 'ground_truth']:    
        
        if return_dict[key] is not None:
            return_dict['preprocess_' + key] = [TF.crop(i, 
                *preprocess_crop_location, *preprocess_crop_size) 
                    for i in return_dict[key]]
        
        else:
            return_dict['preprocess_' + key] = None

    return return_dict


def smooth_transform(return_dict, preprocess_crop_size):
        
    smooth_resize_size = (preprocess_crop_size[0] // 4, preprocess_crop_size[1] // 4)

    for key in ['pos_anchor', 'neg_anchor']:

        if return_dict[key] is not None:
            
            return_dict['preprocess_' + key] = [TF.resize(i, smooth_resize_size) 
                for i in return_dict['preprocess_' + key]]
            
            return_dict['preprocess_' + key] = [TF.resize(i, preprocess_crop_size) 
                for i in return_dict['preprocess_' + key]]

    return return_dict


def augment_transform(return_dict,
        preprocess_crop_size, model_input_size):

    height = preprocess_crop_size[0]
    width = preprocess_crop_size[1]

    crop_scale = 0.85 + random.random() * 0.15  # [0.85, 1)

    crop_h = np.floor(height * crop_scale)
    crop_w = np.floor(width * crop_scale)
    crop_i = random.randint(0, height - crop_h - 1) 
    crop_j = random.randint(0, width - crop_w - 1)
    
    flip_flag = random.random() > 0.5

    for key in ['img', 'pos_anchor', 'neg_anchor', 'ground_truth']: 

        if return_dict[key] is not None:

            return_dict['model_' + key] = [TF.resized_crop(i, 
                crop_i, crop_j, crop_h, crop_w, model_input_size) 
                    for i in return_dict['preprocess_' + key]]
        else:
            return_dict['model_' + key] = None

    for key in ['img', 'pos_anchor', 'neg_anchor', 'ground_truth']: 

        if return_dict[key] is not None:

            if flip_flag:
                return_dict['model_' + key] = [TF.hflip(i) 
                    for i in return_dict['model_' + key]]

    color_trans = ColorJitter(brightness=0.1,
        contrast=0.1, saturation=0.01, hue=0.01)  # should be small

    return_dict['model_img'] = [color_trans(i) 
        for i in return_dict['model_img']]

    return return_dict


def basic_transform(return_dict, model_input_size):

    for key in ['img', 'pos_anchor', 'neg_anchor', 'ground_truth']:
        
        if return_dict[key] is not None:

            return_dict['model_' + key] = [TF.resize(i, model_input_size) 
                for i in return_dict['preprocess_' + key]]

        else:
            return_dict['model_' + key] = None
    
    return return_dict


def tensor_transform(return_dict):

    for key in ['img', 'pos_anchor', 'neg_anchor', 'ground_truth']:
        
        if return_dict[key] is not None:

            return_dict[key] = [TF.to_tensor(i) 
                for i in return_dict[key]]
            
            return_dict['model_' + key] = [TF.to_tensor(i) 
                for i in return_dict['model_' + key]]
            
            return_dict['preprocess_' + key] = [TF.to_tensor(i) 
                for i in return_dict['preprocess_' + key]]

    return_dict['model_img'] = [TF.normalize(i,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
        for i in return_dict['model_img']]

    return return_dict


def get_datadict(
    img_dir, 
    pos_anchor_dir, 
    neg_anchor_dir, 
    ground_truth_dir, 
    sub_dir_list):

    datadict = {}

    for sub_dir in sub_dir_list:

        print('Loading Data Dict... ', sub_dir)

        datadict[sub_dir] = {}

        keys = ['img']

        if pos_anchor_dir:
            keys.append('pos_anchor')
        if neg_anchor_dir:
            keys.append('neg_anchor')
        if ground_truth_dir:
            keys.append('ground_truth')

        for key in keys:

            if key == 'img':
                working_dir = os.path.join(img_dir, sub_dir)
            elif key == 'pos_anchor':
                working_dir = os.path.join(pos_anchor_dir, sub_dir)
            elif key == 'neg_anchor':
                working_dir = os.path.join(neg_anchor_dir, sub_dir)
            elif key == 'ground_truth':
                working_dir = os.path.join(ground_truth_dir, sub_dir)
            else:
                raise Exception('Key Error.')

            name = [os.path.join(working_dir, i) 
                for i in os.listdir(working_dir)]

            name.sort()

            datadict[sub_dir][key] = name

        #################### File Num Check ##################
        
        img_num = len(datadict[sub_dir]['img'])

        for key in keys:
            assert(len(datadict[sub_dir][key]) == img_num)

    return datadict


class SegmentationDataset(Dataset):
    def __init__(self, datadict, 
                 original_img_size, preprocess_crop_size, 
                 preprocess_crop_location, model_input_size, 
                 data_aug, clip_size=1):
        
        super(SegmentationDataset, self).__init__()

        self.datadict = datadict
        self.data_aug = data_aug
        
        self.original_img_size = original_img_size
        self.preprocess_crop_size = preprocess_crop_size
        self.preprocess_crop_location = preprocess_crop_location
        self.model_input_size = model_input_size
        
        self.clip_size = clip_size

        self.sub_dir_list = [i for i in datadict.keys()]
        self.sub_dir_list.sort()

        self.sub_dir_sizes = [
            len(datadict[i]['img']) - clip_size + 1
                for i in self.sub_dir_list]

        for size in self.sub_dir_sizes:
            assert(size > 0)

    def __len__(self):
        return sum(self.sub_dir_sizes)

    def __getitem__(self, idx):

        for sub_dir_idx, sub_dir in enumerate(self.sub_dir_list):  # Be careful

            if idx < self.sub_dir_sizes[sub_dir_idx]:
                clip_idx = idx
                break
            else:
                idx -= self.sub_dir_sizes[sub_dir_idx]

        return_dict = {}

        for key in ['img', 'pos_anchor', 'neg_anchor', 'ground_truth']:

            if key in self.datadict[sub_dir].keys():

                name = self.datadict[sub_dir][key][clip_idx:clip_idx+self.clip_size]

                return_dict[key] = [Image.open(i) for i in name]

                if key == 'img':

                    return_dict['img_name'] = name

                    return_dict['img_short_name'] = [os.path.join(
                        i.split('/')[-2], i.split('/')[-1]) for i in name]

            else:
                return_dict[key] = None

        ################ Shape Check

        for key in ['img', 'pos_anchor', 'neg_anchor', 'ground_truth']:
            if return_dict[key] is not None:
                for i in return_dict[key]:
                    assert(i.size[1] == self.original_img_size[0])
                    assert(i.size[0] == self.original_img_size[1])

        #####

        return_dict = preprocess_transform(return_dict, 
            self.preprocess_crop_size, self.preprocess_crop_location)

        return_dict = smooth_transform(return_dict, self.preprocess_crop_size)

        if self.data_aug:
            return_dict = augment_transform(return_dict,
                self.preprocess_crop_size, self.model_input_size)
        else:
            return_dict = basic_transform(return_dict, self.model_input_size)

        return_dict = tensor_transform(return_dict)

        ################ 

        for key in ['img', 'pos_anchor', 'neg_anchor', 'ground_truth']:

            if return_dict[key] is not None:

                return_dict[key] = torch.cat(
                    [i.unsqueeze(0) for i in return_dict[key]], 0)

                return_dict['model_' + key] = torch.cat(
                    [i.unsqueeze(0) for i in return_dict['model_' + key]], 0)

                return_dict['preprocess_' + key] = torch.cat(
                    [i.unsqueeze(0) for i in return_dict['preprocess_' + key]], 0)

        ########## Shape Check ##########

        for key in ['img', 'pos_anchor', 'neg_anchor', 'ground_truth']:

            if return_dict[key] is not None:

                assert(return_dict[key].shape[0] == self.clip_size)
                assert(return_dict['model_' + key].shape[0] == self.clip_size)
                assert(return_dict['preprocess_' + key].shape[0] == self.clip_size)

                assert(return_dict[key].shape[2] == self.original_img_size[0])
                assert(return_dict[key].shape[3] == self.original_img_size[1])

                assert(return_dict['model_' + key].shape[2] == self.model_input_size[0])
                assert(return_dict['model_' + key].shape[3] == self.model_input_size[1])
            
                assert(return_dict['preprocess_' + key].shape[2] == self.preprocess_crop_size[0])
                assert(return_dict['preprocess_' + key].shape[3] == self.preprocess_crop_size[1])

            else:
                return_dict.pop(key)
                return_dict.pop('model_' + key)
                return_dict.pop('preprocess_' + key)

        return return_dict

