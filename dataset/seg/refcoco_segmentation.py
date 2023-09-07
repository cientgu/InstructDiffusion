# --------------------------------------------------------
# InstructDiffusion
# Based on instruct-pix2pix (https://github.com/timothybrooks/instruct-pix2pix)
# Modified by Binxin Yang (tennyson@mail.ustc.edu.cn)
# --------------------------------------------------------

from __future__ import annotations

import os
import random
import copy
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

from dataset.seg.refcoco import REFER


class RefCOCODataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        transparency: float = 0.0,
        test: bool = False,
    ):
        assert split in ("train", "val", "test")
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.G_ref_dataset=REFER(data_root=path)
        self.IMAGE_DIR = os.path.join(path, 'images/train2014')
        self.list_ref=self.G_ref_dataset.getRefIds(split=split)
        self.transparency = transparency
        self.test = test

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

    def __len__(self) -> int:
        return len(self.list_ref)

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
        image = Image.fromarray(image).resize((self.min_resize_res, self.min_resize_res), resample=Image.Resampling.LANCZOS)
        image = np.asarray(image, dtype=np.uint8)
        label = Image.fromarray(label).resize((self.min_resize_res, self.min_resize_res), resample=Image.Resampling.NEAREST)
        label = np.asarray(label, dtype=np.int64)
        return image, label

    def __getitem__(self, i: int) -> dict[str, Any]:
        
        ref_ids = self.list_ref[i] 
        ref = self.G_ref_dataset.loadRefs(ref_ids)[0]
        sentences = random.choice(ref['sentences'])['sent']

        prompt = random.choice(self.seg_diverse_prompt_list)

        color = random.choice(self.color_list)
        color_name = color[0]
        prompt = prompt.format(color=color_name.lower(), object=sentences.lower())
            
        R, G, B = color[3].split(",")
        R = int(R)
        G = int(G)
        B = int(B)

        image_name = self.G_ref_dataset.loadImgs(ref['image_id'])[0]['file_name']
        image_path = os.path.join(self.IMAGE_DIR,image_name)
        mask = self.G_ref_dataset.getMask(ref=ref)['mask']

        image = Image.open(image_path).convert("RGB")
        image = np.asarray(image)

        image, mask = self._augmentation_new(image,mask)

        mask = (mask == 1)

        image_0 = Image.fromarray(image)
        image_1 = copy.deepcopy(image)
        image_1[:,:,0][mask]=self.transparency*image_1[:,:,0][mask]+(1-self.transparency)*R
        image_1[:,:,1][mask]=self.transparency*image_1[:,:,1][mask]+(1-self.transparency)*G
        image_1[:,:,2][mask]=self.transparency*image_1[:,:,2][mask]+(1-self.transparency)*B
        image_1 = Image.fromarray(image_1)

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        
        
        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        mask = torch.tensor(mask).float()
        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)
        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))