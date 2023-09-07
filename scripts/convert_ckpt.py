# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zigang Geng (zigang@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import annotations

import sys
import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/instruct_diffusion.yaml", type=str)
    parser.add_argument("--ema-ckpt", default="logs/instruct_diffusion/checkpoints/ckpt_epoch_200/state.pth", type=str)
    parser.add_argument("--vae-ckpt", default="stable_diffusion/models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt", type=str)
    parser.add_argument("--out-ckpt", default="checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt", type=str)

    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    model = instantiate_from_config(config.model)

    ema_ckpt = torch.load(args.ema_ckpt, map_location="cpu")
    all_keys = [key for key, value in model.named_parameters()]
    all_keys_rmv = [key.replace('.','') for key in all_keys]
    new_ema_ckpt = {}
    for k, v in ema_ckpt['model_ema'].items():
        try:
            k_index = all_keys_rmv.index(k)
            new_ema_ckpt[all_keys[k_index]] = v
        except:
            print(k+' is not in the list.')

    vae_ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    for k, v in vae_ckpt['state_dict'].items():
        if k not in new_ema_ckpt and k in all_keys:
            new_ema_ckpt[k] = v

    checkpoint = {'state_dict': new_ema_ckpt}
    with open(args.out_ckpt, 'wb') as f:
        torch.save(checkpoint, f)
        f.flush()
    print('Converted successfully, the new checkpoint has been saved to ' + str(args.out_ckpt))