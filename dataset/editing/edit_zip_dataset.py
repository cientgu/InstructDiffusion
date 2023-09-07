# --------------------------------------------------------
# InstructDiffusion
# Based on instruct-pix2pix (https://github.com/timothybrooks/instruct-pix2pix)
# Modified by Tiankai Hang (tkhang@seu.edu.cn)
# --------------------------------------------------------

from __future__ import annotations

import os
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange
import PIL
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

import random

from dataset.utils.zip_manager import MultipleZipManager


if hasattr(Image, "Resampling"):
    # deprecated in pillow >= 10.0.0
    RESAMPLING_METHOD = Image.Resampling.LANCZOS
else:
    RESAMPLING_METHOD = Image.LANCZOS


class FilteredIP2PDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        zip_start_index: int = 0,
        zip_end_index: int = 30,
        instruct: bool = False,
        max_num_images = None,
        sample_weight: float = 1.0,
        reverse_version: bool = False,
        **kwargs
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.instruct = instruct

        zip_list = []
        for i in range(zip_start_index, zip_end_index):
            name = "shard-"+str(i).zfill(2)+'.zip'
            zip_list.append(os.path.join(self.path, name))

        self.image_dataset = MultipleZipManager(zip_list, 'image', sync=True)   # sync=True is faster

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

        if max_num_images is not None and max_num_images > 0:
            self.seeds = self.seeds[:min(max_num_images, len(self.seeds))]

        # flatten seeds
        self.seeds = [(name, seed) for name, seeds in self.seeds for seed in seeds]
        self.sample_weight = sample_weight
        
        while True:
            try:
                with open('filtered_ids_ip2p.json') as json_file:
                    filtered_ids = json.load(json_file)
                break
            except:
                # download json file from url
                if reverse_version:
                    os.system('wget https://github.com/TiankaiHang/storage/releases/download/readout/filtered_ids_ip2p.json')
                else:
                    os.system("wget https://github.com/TiankaiHang/storage/releases/download/readout/filtered-ip2p-thres5.5-0.5.json -O filtered_ids_ip2p.json")
        
        print("seeds:", len(self.seeds))
        # self.seeds = [seed for seed in self.seeds if seed[1] in filtered_ids]
        # faster
        # self.seeds = list(filter(lambda seed: seed[1] in filtered_ids, self.seeds))
        # to numpy and faster in parallel
        # import pdb; pdb.set_trace()
        _seeds = [f"{a}/{b}" for a, b in self.seeds]
        self.seeds = np.array(self.seeds)
        _seeds = np.array(_seeds)
        self.seeds = self.seeds[np.isin(_seeds, filtered_ids)]
        self.seeds = self.seeds.tolist()

        self.return_add_kwargs = kwargs.get("return_add_kwargs", False)

    def __len__(self) -> int:
        return int(len(self.seeds) * self.sample_weight)

    def __getitem__(self, i: int) -> dict[str, Any]:
        # name, seeds = self.seeds[i]
        if self.sample_weight >= 1:
            i = i % len(self.seeds)
        else:
            remainder = math.ceil(i / self.sample_weight - int(i / self.sample_weight))
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1 + remainder)

        name, seed = self.seeds[i]
        propt_name = name + "/prompt.json"
        if not self.image_dataset.managers[self.image_dataset.mapping[propt_name]]._init:
            self.image_dataset.managers[self.image_dataset.mapping[propt_name]].initialize(close=False)
        # propt_name = name + "/prompt.json"
        byteflow = self.image_dataset.managers[self.image_dataset.mapping[propt_name]].zip_fd.read(propt_name)
        texts = json.loads(byteflow.decode('utf-8'))
        prompt = texts["edit"]
        if self.instruct:
            prompt = "Image Editing: " + prompt

        text_input = texts["input"]
        text_output = texts["output"]

        # image_0 = Image.open(propt_dir.joinpath(f"{seed}_0.jpg"))
        # image_1 = Image.open(propt_dir.joinpath(f"{seed}_1.jpg"))
        image_0 = self.image_dataset.get(name+f"/{seed}_0.jpg")
        image_1 = self.image_dataset.get(name+f"/{seed}_1.jpg")

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), RESAMPLING_METHOD)
        image_1 = image_1.resize((reize_res, reize_res), RESAMPLING_METHOD)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        if self.return_add_kwargs:
            add_kwargs = dict(
                name=name,
                seed=seed,
                text_input=text_input,
                text_output=text_output,
            )
        else:
            add_kwargs = {}

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt), **add_kwargs)


class GIERDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        zip_start_index: int = 0,
        zip_end_index: int = 30,
        sample_weight: float = 1.0,
        instruct: bool = False,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.instruct = instruct

        # self.meta = torch.load(Path(self.path, "GIER.json"), map_location="cpu")
        # load json file
        with open(Path(self.path, "GIER_new.json")) as json_file:
            self.meta = json.load(json_file)

        print(f"||||||||||||||||||||||||||||| \n Loaded {len(self.meta)} images from json file")

        input_does_not_exist = []
        output_does_not_exist = []
        # filter out out images that do not exist
        if not os.path.exists(os.path.join(self.path, "filtered_meta_new.pt")):
            filtered_meta = []
            for i in tqdm(range(len(self.meta))):
                input_path = os.path.join(self.path, "warped", self.meta[i]["input"])
                output_path = os.path.join(self.path, "warped", self.meta[i]["output"])

                if not os.path.exists(input_path):
                    input_path = os.path.join(self.path, "images", self.meta[i]["input"])
                    if not os.path.exists(input_path):
                        input_does_not_exist.append(input_path)
                
                if not os.path.exists(output_path):
                    output_path = os.path.join(self.path, "images", self.meta[i]["output"])
                    if not os.path.exists(output_path):
                        output_does_not_exist.append(output_path)
                
                if os.path.exists(input_path) and os.path.exists(output_path):
                    filtered_meta.append(
                        dict(
                            input=input_path,
                            output=output_path,
                            prompts=self.meta[i]["prompts"],
                        )
                    )
                else:
                    print(f"\n {input_path} or {output_path} does not exist")
            torch.save(filtered_meta, os.path.join(self.path, "filtered_meta_new.pt"))
        else:
            filtered_meta = torch.load(os.path.join(self.path, "filtered_meta_new.pt"), map_location="cpu")
        
        self.meta = filtered_meta
        print(f"||||||||||||||||||||||||||||| \n Filtered {len(self.meta)} images")
        for i in range(len(self.meta)):
            self.meta[i]['input'] = self.meta[i]['input'].replace('/mnt/external/datasets/GIER_editing_data/', self.path)
            self.meta[i]['output'] = self.meta[i]['output'].replace('/mnt/external/datasets/GIER_editing_data/', self.path)

        # write input_does_not_exist and output_does_not_exist to file
        with open(Path(self.path, f"input_does_not_exist.txt"), "w") as f:
            for item in input_does_not_exist:
                f.write("%s\n" % item)
        with open(Path(self.path, f"output_does_not_exist.txt"), "w") as f:
            for item in output_does_not_exist:
                f.write("%s\n" % item)
        
        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val":   (splits[0], splits[0] + splits[1]),
            "test":  (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.meta))
        idx_1 = math.floor(split_1 * len(self.meta))
        
        self.meta = self.meta[idx_0:idx_1]
        self.sample_weight = sample_weight
        print('original GIER', len(self.meta))

    def __len__(self) -> int:
        return int(len(self.meta) * self.sample_weight)

    def __getitem__(self, i: int) -> dict[str, Any]:
        if self.sample_weight >= 1:
            i = i % len(self.meta)
        else:
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1)

        # prompt = self.meta[i]["prompts"]
        prompt = random.choice(self.meta[i]["prompts"])
        try:
            image_0 = Image.open(self.meta[i]["input"]).convert("RGB")
            image_1 = Image.open(self.meta[i]["output"]).convert("RGB")
        except PIL.UnidentifiedImageError:
            print(f"\n {self.meta[i]['input']} or {self.meta[i]['output']} is not a valid image")
            i = random.randint(0, len(self.meta) - 1)
            return self.__getitem__(i)

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), RESAMPLING_METHOD)
        image_1 = image_1.resize((reize_res, reize_res), RESAMPLING_METHOD)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        if self.instruct:
            prompt = "Image Editing: " + prompt

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))


class GQAInpaintDataset(Dataset):
    r"""
    shoud download and unzip the data first

    ```
    mkdir -p ../datasets
    cd ../datasets

    # if file exists, then skip
    if [ ! -f "gqa-inpaint.zip" ]; then
        sudo azcopy copy "https://bingdatawu2.blob.core.windows.net/genrecog/private/t-thang/gqa-inpaint.zip${TOKEN}" .
        unzip gqa-inpaint.zip -d gqa-inpaint > /dev/null
    fi

    if [ ! -f "images.zip" ]; then
        sudo azcopy copy "https://bingdatawu2.blob.core.windows.net/genrecog/private/t-thang/images.zip${TOKEN}" .
        unzip images.zip > /dev/null
    fi
    ```
    
    """
    def __init__(self, **kwargs):
        # load from json ../datasets/gqa-inpaint/meta_info.json
        self.path = kwargs.get("path", "../datasets/gqa-inpaint")
        self.instruct = kwargs.get("instruct", False)
        with open(self.path + "/meta_info.json", "r") as f:
            self.meta_info = json.load(f)

        self.min_resize_res = kwargs.get("min_resize_res", 256)
        self.max_resize_res = kwargs.get("max_resize_res", 256)
        self.crop_res = kwargs.get("crop_res", 256)

        self.flip_prob = kwargs.get("flip_prob", 0.5)

    def __len__(self):
        return len(self.meta_info)

    def __getitem__(self, i):
        item = self.meta_info[i]
        src_img = Image.open(item["source_image_path"].replace("../datasets", self.path)).convert("RGB")
        tgt_img = Image.open(item["target_image_path"].replace("../datasets/gqa-inpaint", self.path)).convert("RGB")

        image_0 = src_img
        image_1 = tgt_img

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), RESAMPLING_METHOD)
        image_1 = image_1.resize((reize_res, reize_res), RESAMPLING_METHOD)
        instruction = item["instruction"]
        if self.instruct:
            instruction = "Image Editing: " + instruction
        # return image_0, image_1, instruction

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=instruction))


class MagicBrushDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        zip_start_index: int = 0,
        zip_end_index: int = 30,
        len_dataset: int = -1,
        instruct: bool = False,
        sample_weight: float = 1.0,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.instruct = instruct
        self.sample_weight = sample_weight

        self.meta_path = os.path.join(self.path, "magic_train.json")
        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)

    def __len__(self) -> int:
        return int(len(self.meta) * self.sample_weight)

    def __getitem__(self, i: int) -> dict[str, Any]:
        if self.sample_weight >= 1:
            i = i % len(self.meta)
        else:
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1)

        item = self.meta[i]
        try:
            image_0 = Image.open(os.path.join(self.path, item["input"])).convert("RGB")
            image_1 = Image.open(os.path.join(self.path, item["edited"])).convert("RGB")
        except (PIL.UnidentifiedImageError, FileNotFoundError):
            print(f"\n {self.path}/{item['input']} or {self.path}/{item['edited']} is not a valid image")
            i = random.randint(0, len(self.meta) - 1)
            return self.__getitem__(i)
        prompt = item["instruction"]

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), RESAMPLING_METHOD)
        image_1 = image_1.resize((reize_res, reize_res), RESAMPLING_METHOD)

        if self.instruct:
            prompt = "Image Editing: " + prompt
        # return image_0, image_1, prompt

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))


class IEIWDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        zip_start_index: int = 0,
        zip_end_index: int = 30,
        sample_weight: float = 1.0,
        instruct: bool = False,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.instruct = instruct

        self.meta_path = os.path.join(self.path, "meta_infov1.json")
        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)
        self.sample_weight = sample_weight
        print('original synthetic', len(self.meta))

    def __len__(self) -> int:
        return int(len(self.meta) * self.sample_weight)

    def __getitem__(self, i: int) -> dict[str, Any]:
        if self.sample_weight >= 1:
            i = i % len(self.meta)
        else:
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1)

        item = self.meta[i]
        item['input'] = item['input'].replace('/mnt/external/tmp/2023/06/11/', self.path)
        item['edited'] = item['edited'].replace('/mnt/external/tmp/2023/06/11/', self.path)
        try:
            image_0 = Image.open(item["input"]).convert("RGB")
            image_1 = Image.open(item["edited"]).convert("RGB")
        except (PIL.UnidentifiedImageError, FileNotFoundError):
            print(f"\n {item['input']} or {item['edited']} is not a valid image")
            i = random.randint(0, len(self.meta) - 1)
            return self.__getitem__(i)
        prompt = item["instruction"]

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), RESAMPLING_METHOD)
        image_1 = image_1.resize((reize_res, reize_res), RESAMPLING_METHOD)
        if self.instruct:
            prompt = "Image Editing: " + prompt
        # return image_0, image_1, prompt

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))


