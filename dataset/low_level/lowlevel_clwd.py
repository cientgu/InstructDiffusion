# --------------------------------------------------------
# InstructDiffusion
# Based on instruct-pix2pix (https://github.com/timothybrooks/instruct-pix2pix)
# Modified by Chen Li (edward82@stu.xjtu.edu.cn)
# --------------------------------------------------------

import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random
import cv2
from PIL import Image
import torchvision


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class CLWD(Dataset):
    def __init__(self, path, split="train", size=256, interpolation="pil_lanczos", 
        flip_prob=0.5, sample_weight=1.0, instruct=False):
        super(CLWD, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(path, split, 'Watermarked_image')))
        tar_files = sorted(os.listdir(os.path.join(path, split, 'Watermark_free_image')))

        self.inp_filenames = [os.path.join(path, split, 'Watermarked_image', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(path, split, 'Watermark_free_image', x) for x in tar_files if is_image_file(x)]

        self.size = size
        self.flip_prob = flip_prob
        self.sample_weight = sample_weight
        self.instruct = instruct
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.interpolation = {
            "cv_nearest": cv2.INTER_NEAREST,
            "cv_bilinear": cv2.INTER_LINEAR,
            "cv_bicubic": cv2.INTER_CUBIC,
            "cv_area": cv2.INTER_AREA,
            "cv_lanczos": cv2.INTER_LANCZOS4,
            "pil_nearest": Image.NEAREST,
            "pil_bilinear": Image.BILINEAR,
            "pil_bicubic": Image.BICUBIC,
            "pil_box": Image.BOX,
            "pil_hamming": Image.HAMMING,
            "pil_lanczos": Image.LANCZOS,
        }[interpolation]

        prompt_path='dataset/prompt/prompt_dewatermark.txt'
        self.prompt_list=[]
        with open(prompt_path) as f:
            line=f.readline()
            while line:
                line=line.strip('\n')
                self.prompt_list.append(line)
                line=f.readline()
        
        print(f"CLWD has {len(self)} samples!!")

    def __len__(self):
        return int(self.sizex * self.sample_weight)

    def __getitem__(self, index):
        if self.sample_weight >= 1:
            index_ = index % self.sizex
        else:
            index_ = int(index / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1)

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)

        width, height = inp_img.size
        tar_width, tar_height = tar_img.size
        assert tar_width == width and tar_height == height, "Input and target image mismatch"
        aspect_ratio = float(width) / float(height)
        if width < height:  
            new_width = self.size  
            new_height = int(self.size  / aspect_ratio)  
        else:  
            new_height = self.size   
            new_width = int(self.size * aspect_ratio) 
        inp_img = inp_img.resize((new_width, new_height), self.interpolation)
        tar_img = tar_img.resize((new_width, new_height), self.interpolation)

        inp_img = np.array(inp_img).astype(np.float32).transpose(2, 0, 1)
        inp_img_tensor = torch.tensor((inp_img / 127.5 - 1.0).astype(np.float32))
        tar_img = np.array(tar_img).astype(np.float32).transpose(2, 0, 1)
        tar_img_tensor = torch.tensor((tar_img / 127.5 - 1.0).astype(np.float32))
        crop = torchvision.transforms.RandomCrop(self.size)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((inp_img_tensor, tar_img_tensor)))).chunk(2)

        prompt = random.choice(self.prompt_list)
        if self.instruct:
            prompt = "Watermark Removal: " + prompt

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))