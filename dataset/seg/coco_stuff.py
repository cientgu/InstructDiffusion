# --------------------------------------------------------
# InstructDiffusion
# Based on instruct-pix2pix (https://github.com/timothybrooks/instruct-pix2pix)
# Modified by Binxin Yang (tennyson@mail.ustc.edu.cn)
# --------------------------------------------------------

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
import cv2
import os
import random
import copy
from glob import glob


class COCOStuffDataset(Dataset):
    def __init__(
        self,
        path: str,
        path_edit: str = "None",
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        crop_res: int = 256,
        flip_prob: float = 0.0,
        transparency: float = 0,
        batch_size: int = 10,
        empty_percentage: float = 0,
    ):
        assert split in ("train2017", "val2017")
        assert sum(splits) == 1
        self.split = split
        self.path = path
        self.path_edit = path_edit
        self.batch_size = batch_size
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.empty_percentage = empty_percentage
        self.transparency = transparency
        if self.split in ["train2017", "val2017"]:
            file_list = sorted(glob(os.path.join(self.path, "images", self.split, "*.jpg")))
            assert len(file_list) > 0, "{} has no image".format(
                os.path.join(self.path, "images", self.split)
            )
            file_list = [f.split("/")[-1].replace(".jpg", "") for f in file_list]
            self.files = file_list

        else:
            raise ValueError("Invalid split name: {}".format(self.split))

        seg_diverse_prompt_path = 'dataset/prompt/prompt_seg.txt'
        self.seg_diverse_prompt_list=[]
        with open(seg_diverse_prompt_path) as f:
            line=f.readline()
            while line:
                line=line.strip('\n')
                self.seg_diverse_prompt_list.append(line)
                line=f.readline()

        color_list_file_path='dataset/prompt/color_list_train_small.txt'
        self.color_list=[]
        with open(color_list_file_path) as f:
            line = f.readline()
            while line:
                line_split = line.strip('\n').split(" ")
                if len(line_split)>1:
                    temp = []
                    for i in range(4):
                        temp.append(line_split[i])
                    self.color_list.append(temp)
                line = f.readline()

        coco_label_list_path = self.path + '/labels.txt'
        self.label_dict={}
        with open(coco_label_list_path) as f:
            line = f.readline()
            while line:
                line_split = line.strip('\n').split(": ")
                self.label_dict[int(line_split[0])]=line_split[1]
                line = f.readline()

    def __len__(self) -> int:
        length=len(self.files)
        return length
    
    def _augmentation_new(self, image, label):

        # Cropping
        h, w = label.shape
        if h > w:
            start_h = random.randint(0, h - w)
            end_h = start_h + w
            image = image[start_h:end_h]
            label = label[start_h:end_h]
        elif h < w:
            start_w = random.randint(0, w - h)
            end_w = start_w + h
            image = image[:, start_w:end_w]
            label = label[:, start_w:end_w]
        else:
            pass
        image = Image.fromarray(image).resize((self.crop_res, self.crop_res), resample=Image.Resampling.LANCZOS)
        image = np.asarray(image, dtype=np.uint8)
        label = Image.fromarray(label).resize((self.crop_res, self.crop_res), resample=Image.Resampling.NEAREST)
        label = np.asarray(label, dtype=np.int64)
        return image, label

    def __getitem__(self, i):

        image_id = self.files[i]
        img_path = os.path.join(self.path, "images", self.split, image_id + ".jpg")
        mask_path = os.path.join(self.path, "annotations", self.split, image_id + ".png")

        label = Image.open(mask_path).convert("L")
        image = Image.open(img_path).convert("RGB")
        label = np.asarray(label)
        image = np.asarray(image)
        image, label = self._augmentation_new(image,label)

        label_list = np.unique(label)
        label_list = list(label_list)
        label_list_rest = [i for i in range(182)]
        for item in label_list_rest:
            if item in label_list:
                label_list_rest.remove(item)
        if 255 in label_list:
            label_list.remove(255)
        if len(label_list)!=0:
            label_idx = random.choice(label_list)
            if random.uniform(0, 1) < self.empty_percentage:
                label_idx = random.choice(label_list_rest)

            class_name = self.label_dict[label_idx+1]

            prompt = random.choice(self.seg_diverse_prompt_list)
            color = random.choice(self.color_list)
            color_name = color[0]
            prompt = prompt.format(color=color_name.lower(), object=class_name.lower())
            R, G, B = color[3].split(",")
            R = int(R)
            G = int(G)
            B = int(B)
        else:
            label_idx = 200
            prompt = "leave the picture as it is."
        mask = (label==label_idx)
        image_0 = Image.fromarray(image)
        image_1 = copy.deepcopy(image)

        if len(label_list)!=0:
            image_1[:,:,0][mask]=self.transparency*image_1[:,:,0][mask]+(1-self.transparency)*R
            image_1[:,:,1][mask]=self.transparency*image_1[:,:,1][mask]+(1-self.transparency)*G
            image_1[:,:,2][mask]=self.transparency*image_1[:,:,2][mask]+(1-self.transparency)*B

        image_1 = Image.fromarray(image_1)
        # return image_0, image_1, prompt

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        mask = torch.tensor(mask).float()
        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)
        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))