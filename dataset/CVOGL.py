import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import random
import logging

logger = logging.getLogger(__name__)

class LimitedFoV(object):

    def __init__(self, fov=360.):
        self.fov = fov

    def __call__(self, x):
        angle = random.randint(0, 359)
        rotate_index = int(angle / 360. * x.shape[2])
        fov_index = int(self.fov / 360. * x.shape[2])
        if rotate_index > 0:
            img_shift = torch.zeros(x.shape)
            img_shift[:,:, :rotate_index] = x[:,:, -rotate_index:]
            img_shift[:,:, rotate_index:] = x[:,:, :(x.shape[2] - rotate_index)]
        else:
            img_shift = x

        return img_shift[:,:,:fov_index]


def input_transform_fov(size, fov):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        LimitedFoV(fov=fov),
    ])

def input_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])


# Same loader from CVOGL, modified for pytorch
class CVOGL(torch.utils.data.Dataset):
    def __init__(self, mode='', root='/raid/lingxingtao/dataset/CVOGL/', same_area=True, print_bool=False, args=None,
                 sat_size=[1024, 1024], grd_size=[256, 256]):
        super(CVOGL, self).__init__()
        # SEPERATOR: init
        self.args = args
        self.root = root
        self.mode = mode
        self.sat_size = sat_size
        self.grd_size = grd_size
        self.data_name = args.data_name # avoid to pass a new parameter# avoid to pass a new parameter in code
        assert(self.data_name in ['CVOGL_DroneAerial', 'CVOGL_SVI'], 
               f"[ERROR] Unknown property '{self.data_name}' of 'args.data_name': property 'args.data_name' should be in: 'CVOGL_DroneAerial', 'CVOGL_SVI'!")
        
        self.sat_size_default = [1024, 1024]
        # DroneAerial [256, 256], SVI [512, 256]
        if self.data_name == 'CVOGL_DroneAerial':
            self.grd_size_default = [256, 256]
        elif self.data_name == 'CVOGL_SVI':
            self.grd_size_default = [512, 256]

        assert(mode in ['train', 'test_query', 'test_reference'], 
               f"[ERROR] Unknown property '{mode}' of 'mode': property 'mode' should be in: 'train', 'test_query', 'test_reference'!")
        if print_bool:
            print(self.sat_size, self.grd_size)

        # QUESTION: 这是什么? from VIGOR
        self.sat_ori_size = [640, 640]
        self.grd_ori_size = [1024, 2048]

        # SEPERATOR: transforms
        if args.fov != 0:
            self.transform_query = input_transform_fov(size=self.grd_size,fov=args.fov)
        else:
            self.transform_query = input_transform(size=self.grd_size)
        self.transform_reference = input_transform(size=self.sat_size)
        self.to_tensor = transforms.ToTensor()
        # only for VIGOR
        self.same_area = same_area
        
        # SEPERATOR: get data file
        # load list
        self.train_list_fname = os.path.join(self.root, self.data_name, f'{self.data_name}_train.pth')
        self.test_list_fname = os.path.join(self.root, self.data_name, f'{self.data_name}_test.pth')
        self.sat_root = os.path.join(self.root, self.data_name, 'satellite')
        self.grd_root = os.path.join(self.root, self.data_name, 'query')

        # SEPERATOR: train data
        self.__cur_id = 0  # for training
        self.id_list = [] # [(sat_img, grd_img, id)]
        self.id_idx_list = [] # [idx]
        self.id_data_list = [] # [(sat_img, grd_img, gt_box, click_xy)]
        self.train_list = torch.load(self.train_list_fname)
        for data in self.train_list:
            id, grd_img, sat_img, _, click_xy, gt_box, _, _ = data
            sat_img = os.path.join(self.sat_root, sat_img)
            grd_img = os.path.join(self.grd_root, grd_img)
            self.id_data_list.append((sat_img, grd_img, gt_box, click_xy)) # use tuple instead of list
            # keep same
            self.id_list.append((sat_img, grd_img, id)) # use tuple instead of list
            self.id_idx_list.append(idx)
            idx += 1
        self.train_data_size = len(self.train_sat_list)

        if print_bool:
            logger.info("load train set: %s", self.train_list_fname)
            logger.info("load train set size: %d", self.train_data_size)
            print(f'[INFO] dataset.CVOGL::__init__: load train set: {self.train_list_fname}')
            print(f'[INFO] dataset.CVOGL::__init__: load train set: {self.train_data_size}')

        # SEPERATOR: test data
        self.__cur_test_id = 0  # for training
        self.id_test_list = []
        self.id_test_idx_list = []
        self.id_test_data_list = []
        for data in self.test_list:
            id, grd_img, sat_img, _, click_xy, gt_box, _, _ = data
            sat_img = os.path.join(self.sat_root, sat_img)
            grd_img = os.path.join(self.grd_root, grd_img)
            self.id_test_data_list.append((sat_img, grd_img, gt_box, click_xy)) # use tuple instead of list
            # keep same
            self.id_test_list.append((sat_img, grd_img, id)) # use tuple instead of list
            self.id_test_idx_list.append(idx)
            idx += 1
        self.test_data_size = len(self.id_test_list)
        if print_bool:
            logger.info("load test set: %s", self.test_list_fname)
            logger.info("load test set size: %d", self.test_data_size)
            print(f'[INFO] dataset.CVOGL::__init__: load test set: {self.test_list_fname}')
            print(f'[INFO] dataset.CVOGL::__init__: load test set: {self.test_data_size}')

    def __getitem__(self, index, debug=False):
        if self.mode == 'train':
            idx = index % len(self.id_idx_list)
            img_query = Image.open(self.root + self.id_list[idx][1]).convert('RGB')
            img_reference = Image.open(self.root + self.id_list[idx][0]).convert('RGB')

            img_query = self.transform_query(img_query)
            img_reference = self.transform_reference(img_reference)
            return img_query, img_reference, torch.tensor(idx), torch.tensor(idx)

        elif 'test_reference' in self.mode:
            img_reference = Image.open(self.root + self.id_test_list[index][0]).convert('RGB')
            img_reference = self.transform_reference(img_reference)
            return img_reference, torch.tensor(index)

        elif 'test_query' in self.mode:
            img_query = Image.open(self.root + self.id_test_list[index][1]).convert('RGB')
            img_query = self.transform_query(img_query)
            return img_query, torch.tensor(index), torch.tensor(index)
        else:
            logger.error("Unknown mode '%s', should be 'train', 'test_reference', 'test_query'", self.mode)
            print(f"[ERROR]: '{self.mode}' mode not implemented!!")
            raise NotImplementedError(f"'{self.mode}' mode not implemented!!")

    def __len__(self):
        if self.mode == 'train':
            return len(self.id_idx_list)
        elif 'test_reference' in self.mode:
            return len(self.id_test_list)
        elif 'test_query' in self.mode:
            return len(self.id_test_list)
        else:
            logger.error("Unknown mode '%s', should be 'train', 'test_reference', 'test_query'", self.mode)
            print(f"[ERROR]: '{self.mode}' mode not implemented!!")
            raise NotImplementedError(f"'{self.mode}' mode not implemented!!")
