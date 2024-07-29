import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import albumentations
from shapely.geometry import Polygon

import os
import random
import logging

logger = logging.getLogger(__name__)


# SEPERATOR: DetGeo utils
class DatasetNotFoundError(Exception):
    pass


class MyAugment:
    def __init__(self) -> None:
        self.transform = albumentations.Compose([
            albumentations.Blur(p=0.01),
            albumentations.MedianBlur(p=0.01),
            albumentations.ToGray(p=0.01),
            albumentations.CLAHE(p=0.01),
            albumentations.RandomBrightnessContrast(p=0.0),
            albumentations.RandomGamma(p=0.0),
            albumentations.ImageCompression(quality_lower=75, p=0.0)])

    def augment_hsv(self, im, hgain=0.5, sgain=0.5, vgain=0.5):
        # HSV color-space augmentation
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * \
                [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))
            dtype = im.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(
                sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=im)  # no return needed

    def __call__(self, img, bbox):
        imgh, imgw, _ = img.shape
        x, y, w, h = (bbox[0]+bbox[2])/2/imgw, (bbox[1]+bbox[3]) / \
            2/imgh, (bbox[2]-bbox[0])/imgw, (bbox[3]-bbox[1])/imgh
        img = self.transform(image=img)['image']
        # self.augment_hsv(img)
        # Flip up-down
        if random.random() < 0.5:
            img = np.flipud(img)
            y = 1-y

        # Flip left-right
        if random.random() < 0.5:
            img = np.fliplr(img)
            x = 1-x
        #
        new_imgh, new_imgw, _ = img.shape
        assert new_imgh == imgh, new_imgw == imgw
        x, y, w, h = x*imgw, y*imgh, w*imgw, h*imgh

        # Crop image
        iscropped = False
        if random.random() < 0.5:
            left, top, right, bottom = x-w/2, y-h/2, x+w/2, y+h/2
            if left >= new_imgw/2:
                start_cropped_x = random.randint(0, int(0.15*new_imgw))
                img = img[:, start_cropped_x:, :]
                left, right = left - start_cropped_x, right - start_cropped_x
            if right <= new_imgw/2:
                start_cropped_x = random.randint(int(0.85*new_imgw), new_imgw)
                img = img[:, 0:start_cropped_x, :]
            if top >= new_imgh/2:
                start_cropped_y = random.randint(0, int(0.15*new_imgh))
                img = img[start_cropped_y:, :, :]
                top, bottom = top - start_cropped_y, bottom - start_cropped_y
            if bottom <= new_imgh/2:
                start_cropped_y = random.randint(int(0.85*new_imgh), new_imgh)
                img = img[0:start_cropped_y, :, :]
            cropped_imgh, cropped_imgw, _ = img.shape
            left, top, right, bottom = left/cropped_imgw, top / \
                cropped_imgh, right/cropped_imgw, bottom/cropped_imgh
            if cropped_imgh != new_imgh or cropped_imgw != new_imgw:
                img = cv2.resize(img, (new_imgh, new_imgw))
            new_cropped_imgh, new_cropped_imgw, _ = img.shape
            left, top, right, bottom = left*new_cropped_imgw, top * \
                new_cropped_imgh, right*new_cropped_imgw, bottom*new_cropped_imgh
            x, y, w, h = (left+right)/2, (top+bottom)/2, right-left, bottom-top
            iscropped = True
        # if iscropped:
        #    print((new_imgw, new_imgh))
        #    print((cropped_imgw, cropped_imgh), flush=True)
        #    print('============')
        # print(type(img))
        # draw_bbox = np.array([x-w/2, y-h/2, x+w/2, y+h/2], dtype=int)
        # print(('draw_bbox', iscropped, draw_bbox), flush=True)
        # img_new=draw_rectangle(img, draw_bbox)
        # cv2.imwrite('tmp/'+str(random.randint(0,5000))+"_"+str(iscropped)+".jpg", img_new)

        new_bbox = [(x-w/2), y-h/2, x+w/2, y+h/2]
        # print(bbox)
        # print(new_bbox)
        # print('---end---')
        return img, np.array(new_bbox, dtype=int)


BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, x_max, y_min, y_max = int(bbox[0]), int(
        bbox[2]), int(bbox[1]), int(bbox[3])
    print(bbox, flush=True)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                  color=(255, 0, 0), thickness=2)

    ((text_width, text_height), _) = cv2.getTextSize(
        class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)),
                  (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img

# SEPERATOR: TransGeo fov util


class LimitedFoV(object):

    def __init__(self, fov=360.):
        self.fov = fov

    def __call__(self, x):
        angle = random.randint(0, 359)
        rotate_index = int(angle / 360. * x.shape[2])
        fov_index = int(self.fov / 360. * x.shape[2])
        if rotate_index > 0:
            img_shift = torch.zeros(x.shape)
            img_shift[:, :, :rotate_index] = x[:, :, -rotate_index:]
            img_shift[:, :, rotate_index:] = x[:,
                                               :, :(x.shape[2] - rotate_index)]
        else:
            img_shift = x

        return img_shift[:, :, :fov_index]


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


def detgeo_ref_transform(size: int):
    return albumentations.Compose([   
            albumentations.RandomSizedBBoxSafeCrop(width=size, height=size, erosion_rate=0.2, p=0.2),
	        albumentations.RandomRotate90(p=0.5),
	        albumentations.GaussNoise(p=0.5),
	        albumentations.HueSaturationValue(p=0.3),
	        albumentations.OneOf([
		        albumentations.Blur(p=0.4),
		        albumentations.MedianBlur(p=0.3),
	        ], p=0.5),
	        albumentations.OneOf([
		        albumentations.RandomBrightnessContrast(p=0.4),
		        albumentations.CLAHE(p=0.3),
	        ], p=0.5),
	        albumentations.ToGray(p=0.2),
	        albumentations.RandomGamma(p=0.3),], bbox_params=albumentations.BboxParams(format='pascal_voc'))


# SEPERATOR: TransGeo base dataset
# Same loader from CVOGL, modified for pytorch
class CVOGL(torch.utils.data.Dataset):
    def __init__(self, mode='', root='/data/hushuhan/codes/dataset/CVOGL', same_area=True, print_bool=False, args=None,
                 sat_size=[512, 512], grd_size=[256, 256]):
        super(CVOGL, self).__init__()
        # SEPERATOR: init
        self.args = args
        self.root = root
        self.mode = mode
        assert (mode in ['train', 'test_query', 'test_reference'],
                f"[ERROR] Unknown property '{mode}' of 'mode': property 'mode' should be in: 'train', 'test_query', 'test_reference'!")
        
        # CORE: sat_size 的设置会影响显存占用的大小，sat_size 越大，显存占用越多，性能也会受影响。grd_size同理。
        self.sat_size = sat_size
        self.grd_size = grd_size
        # avoid to pass a new parameter# avoid to pass a new parameter in code
        self.data_name = args.data_name
        assert (self.data_name in ['CVOGL_DroneAerial', 'CVOGL_SVI'],
                f"[ERROR] Unknown property '{self.data_name}' of 'args.data_name': property 'args.data_name' should be in: 'CVOGL_DroneAerial', 'CVOGL_SVI'!")

        self.sat_size_default = [1024, 1024]
        # DroneAerial [256, 256], SVI [512, 256]
        if self.data_name == 'CVOGL_DroneAerial':
            self.grd_size_default = [256, 256]
        elif self.data_name == 'CVOGL_SVI':
            self.grd_size_default = [512, 256]

        if print_bool:
            print(self.sat_size, self.grd_size)

        self.myaugment = MyAugment()

        # SEPERATOR: transforms
        if args.fov != 0:
            self.transform_query = input_transform_fov(
                size=self.grd_size, fov=args.fov)
        else:
            self.transform_query = input_transform(size=self.grd_size)
        self.transform_reference = input_transform(size=self.sat_size)
        self.to_tensor = transforms.ToTensor()

        # SEPERATOR: get data file
        # load list
        self.train_list_fname = os.path.join(self.root, self.data_name, f'{
                                             self.data_name}_train.pth')
        self.test_list_fname = os.path.join(
            self.root, self.data_name, f'{self.data_name}_test.pth')
        self.sat_root = os.path.join(self.root, self.data_name, 'satellite')
        self.grd_root = os.path.join(self.root, self.data_name, 'query')

        # SEPERATOR: train data
        self.__cur_id = 0  # for training
        self.id_list = []  # [(sat_img, grd_img, id)]
        self.id_idx_list = []  # [idx]
        self.id_data_list = []  # [(sat_img, grd_img, gt_box, click_xy)]
        self.train_list = torch.load(self.train_list_fname, weights_only=False)
        idx = 0
        for data in self.train_list:
            id, grd_img, sat_img, _, click_xy, gt_box, _, cls_name = data
            sat_img = os.path.join(self.sat_root, sat_img)
            grd_img = os.path.join(self.grd_root, grd_img)
            # use tuple instead of list
            self.id_data_list.append((sat_img, grd_img, id, gt_box, click_xy, cls_name))
            # keep same
            # use tuple instead of list
            self.id_list.append((sat_img, grd_img, id))
            self.id_idx_list.append(idx)
            idx += 1
        self.train_data_size = len(self.train_list)

        if print_bool:
            logger.info("load train set: %s", self.train_list_fname)
            logger.info("load train set size: %d", self.train_data_size)
            print(f'[INFO] dataset.CVOGL::__init__: load train set: {
                  self.train_list_fname}')
            print(f'[INFO] dataset.CVOGL::__init__: load train set: {
                  self.train_data_size}')

        # SEPERATOR: test data
        self.__cur_test_id = 0  # for training
        self.id_test_list = []
        self.id_test_idx_list = []
        self.id_test_data_list = []
        self.test_list = torch.load(self.test_list_fname, weights_only=False)
        idx = 0
        for data in self.test_list:
            id, grd_img, sat_img, _, click_xy, gt_box, _, cls_name = data
            sat_img = os.path.join(self.sat_root, sat_img)
            grd_img = os.path.join(self.grd_root, grd_img)
            # use tuple instead of list
            self.id_test_data_list.append((sat_img, grd_img, id, gt_box, click_xy, cls_name))
            # keep same
            # use tuple instead of list
            self.id_test_list.append((sat_img, grd_img, id))
            self.id_test_idx_list.append(idx)
            idx += 1
        self.test_data_size = len(self.id_test_list)
        if print_bool:
            logger.info("load test set: %s", self.test_list_fname)
            logger.info("load test set size: %d", self.test_data_size)
            print(f'[INFO] dataset.CVOGL::__init__: load test set: {
                  self.test_list_fname}')
            print(f'[INFO] dataset.CVOGL::__init__: load test set: {
                  self.test_data_size}')

    def __getitem__(self, index, debug=False):
        if self.mode == 'train':
            idx = index % len(self.id_idx_list)
            img_query = Image.open(self.id_data_list[idx][1]).convert('RGB')
            img_reference = Image.open(self.id_data_list[idx][0]).convert('RGB')
            bbox = self.id_data_list[idx][3]
            click_xy = self.id_data_list[idx][4]
            cls_name = self.id_data_list[idx][5]

            ref_transformed = detgeo_ref_transform(self.sat_size[0])(image=np.array(img_reference), bboxes=[list(bbox)+[cls_name]])
            img_reference = Image.fromarray(ref_transformed['image'])
            bbox = ref_transformed['bboxes'][0][0:4]

            img_query = self.transform_query(img_query)
            img_reference = self.transform_reference(img_reference)

            click_hw = (int(click_xy[1]), int(click_xy[0]))
        
            mat_clickhw = np.zeros((self.grd_size[0], self.grd_size[1]), dtype=np.float32)
            click_h = [pow(one-click_hw[0],2) for one in range(self.grd_size[0])]
            click_w = [pow(one-click_hw[1],2) for one in range(self.grd_size[1])]
            norm_hw = pow(self.grd_size[0]*self.grd_size[0] + self.grd_size[1]*self.grd_size[1], 0.5)
            for i in range(self.grd_size[0]):
                for j in range(self.grd_size[1]):
                    tmp_val = 1 - (pow(click_h[i]+click_w[j], 0.5)/norm_hw)
                    mat_clickhw[i, j] = tmp_val * tmp_val
            return img_query, img_reference, torch.tensor(idx), torch.tensor(idx), mat_clickhw, np.array(bbox, dtype=np.float32)

        elif 'test_reference' in self.mode:
            img_reference = Image.open(
                self.id_test_data_list[index][0]).convert('RGB')
            bbox = self.id_test_data_list[index][3]
            img_reference = self.transform_reference(img_reference)
            return img_reference, torch.tensor(index), np.array(bbox, dtype=np.float32)

        elif 'test_query' in self.mode:
            img_query = Image.open(self.id_test_data_list[index][1]).convert('RGB')
            click_xy = self.id_test_data_list[index][4]
            img_query = self.transform_query(img_query)

            click_hw = (int(click_xy[1]), int(click_xy[0]))
        
            mat_clickhw = np.zeros((self.grd_size[0], self.grd_size[1]), dtype=np.float32)
            click_h = [pow(one-click_hw[0],2) for one in range(self.grd_size[0])]
            click_w = [pow(one-click_hw[1],2) for one in range(self.grd_size[1])]
            norm_hw = pow(self.grd_size[0]*self.grd_size[0] + self.grd_size[1]*self.grd_size[1], 0.5)
            for i in range(self.grd_size[0]):
                for j in range(self.grd_size[1]):
                    tmp_val = 1 - (pow(click_h[i]+click_w[j], 0.5)/norm_hw)
                    mat_clickhw[i, j] = tmp_val * tmp_val
            return img_query, torch.tensor(index), torch.tensor(index), mat_clickhw
        else:
            logger.error(
                "Unknown mode '%s', should be 'train', 'test_reference', 'test_query'", self.mode)
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
            logger.error(
                "Unknown mode '%s', should be 'train', 'test_reference', 'test_query'", self.mode)
            print(f"[ERROR]: '{self.mode}' mode not implemented!!")
            raise NotImplementedError(f"'{self.mode}' mode not implemented!!")