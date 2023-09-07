# --------------------------------------------------------
# InstructDiffusion
# Based on instruct-pix2pix (https://github.com/timothybrooks/instruct-pix2pix)
# Modified by Tiankai Hang (tkhang@seu.edu.cn)
# --------------------------------------------------------

import os
import sys
import re

import math
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import autocast
import einops
from einops import rearrange
import gradio as gr
import k_diffusion as K
import requests
from functools import partial
from copy import deepcopy

from PIL import Image, ImageOps
import click

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    model = instantiate_from_config(config.model)

    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if 'state_dict' in pl_sd:
        pl_sd = pl_sd['state_dict']
    m, u = model.load_state_dict(pl_sd, strict=False)

    print(m, u)
    return model


def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content


def get_header():
    content = """
    <div style="text-align: center; max-width: 650px; margin: 0 auto;">
    <div style="
            display: inline-flex;
            gap: 0.8rem;
            font-size: 1.75rem;
            justify-content: center;
            margin-bottom: 10px;
        ">
        <h1 style="font-weight: 900; align-items: center; margin-bottom: 7px; margin-top: 20px;">
        InstructDiffusion 🎨
        </h1>
    </div>
    <div>
        <p style="align-items: center; margin-bottom: 7px;">
        InstructDiffusion, upload a source image and write the instruction to conduct keypoint detection, referring segmentation, and image editing.
        </p>
        <p style="align-items: center; margin-bottom: 7px;">
        Paper is available in <a style="text-decoration: underline;" href="https://gengzigang.github.io/instructdiffusion.github.io/">Arxiv</a>. If you like this demo, please help to ⭐ the <a style="text-decoration: underline;" href="https://github.com/cientgu/InstructDiffusion">Github Repo</a> 😊.
        </p>
    </div>
    </div>
    """
    return content


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], cond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_txt_cond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return 0.5 * (out_img_cond + out_txt_cond) + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_cond - out_txt_cond)


def predict(
        model, model_wrap, 
        model_wrap_cfg,
        null_token, resolution,
        input_img, edit, seed, steps, cfg_text, cfg_image,
        stochastic_steps=0, sampler="euler", additional={}):

    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.cuda.empty_cache()
    
    if isinstance(input_img, str):
        if input_img.startswith("http"):
            input_image = Image.open(requests.get(input_img, stream=True).raw).convert("RGB")
        else:
            input_image = Image.open(input_img).convert("RGB")
        width, height = input_image.size

        factor = resolution / max(width, height)

        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        if hasattr(Image, "Resampling"):
            input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
        else:
            input_image = ImageOps.fit(input_image, (width, height), method=Image.LANCZOS)
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_image = rearrange(input_image, "h w c -> 1 c h w").cuda()

    # if PIL Image
    elif isinstance(input_img, Image.Image):
        input_image = input_img
        width, height = input_image.size
        factor = resolution / max(width, height)
        # factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        if hasattr(Image, "Resampling"):
            input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
        else:
            input_image = ImageOps.fit(input_image, (width, height), method=Image.LANCZOS)
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_image = rearrange(input_image, "h w c -> 1 c h w").cuda()
    elif isinstance(input_img, dict):
        input_image = input_img["image"].convert("RGB")
        width, height = input_image.size

        factor = resolution / max(width, height)

        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        if hasattr(Image, "Resampling"):
            input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
        else:
            input_image = ImageOps.fit(input_image, (width, height), method=Image.LANCZOS)
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_image = rearrange(input_image, "h w c -> 1 c h w").cuda()

    assert input_image is not None
    # print input image size
    print(input_image.shape, factor, width, height)

    with torch.no_grad(), autocast("cuda"):
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([edit])]
        cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

        uncond = {}
        if "txt_embed" in additional:
            uncond["c_crossattn"] = [additional["txt_embed"].cuda().unsqueeze(0)]
        else:
            uncond["c_crossattn"] = [null_token]
        if "img_embed" in additional:
            # uncond["c_concat"] = [additional["img_embed"].cuda()]
            # resize to cond["c_concat"][0]
            uncond["c_concat"] = [additional["img_embed"].cuda()]
            uncond["c_concat"][0] = F.interpolate(uncond["c_concat"][0], size=cond["c_concat"][0].shape[-2:], mode="bilinear", align_corners=False)
        else:
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        sigmas = model_wrap.get_sigmas(steps)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": cfg_text,
            "image_cfg_scale": cfg_image,
        }
        
        if stochastic_steps <= 0:
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            if sampler == "euler":
                z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
            elif sampler == "heun":
                z = K.sampling.sample_heun(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        else:
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[stochastic_steps] + cond["c_concat"][0]
            z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas[stochastic_steps:], extra_args=extra_args)
        
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())

        # input_image to PIL
        input_image = torch.clamp((input_image + 1.0) / 2.0, min=0.0, max=1.0)
        input_image = 255.0 * rearrange(input_image, "1 c h w -> h w c")
        input_image = Image.fromarray(input_image.type(torch.uint8).cpu().numpy())

        return edited_image # , gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)


@click.command()
@click.option("--ckpt", type=str, default="checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt")
def main(ckpt="checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt"):
    css = '''
    .container {max-width: 1150px;margin: auto;padding-top: 1.5rem}
    #image_upload{min-height:400px}
    #image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 400px}
    #mask_radio .gr-form{background:transparent; border: none}
    #word_mask{margin-top: .75em !important}
    #word_mask textarea:disabled{opacity: 0.3}
    .footer {margin-bottom: 45px;margin-top: 35px;text-align: center;border-bottom: 1px solid #e5e5e5}
    .footer>p {font-size: .8rem; display: inline-block; padding: 0 10px;transform: translateY(10px);background: white}
    .dark .footer {border-color: #303030}
    .dark .footer>p {background: #0b0f19}
    .acknowledgments h4{margin: 1.25em 0 .25em 0;font-weight: bold;font-size: 115%}
    #image_upload .touch-none{display: flex}
    @keyframes spin {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }
    #share-btn-container {
        display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
    }
    #share-btn {
        all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;
    }
    #share-btn * {
        all: unset;
    }
    #share-btn-container div:nth-child(-n+2){
        width: auto !important;
        min-height: 0px !important;
    }
    #share-btn-container .wrap {
        display: none !important;
    }
    '''

    config = OmegaConf.load("configs/instruct_diffusion.yaml")

    # ckpt = "checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt"
    
    if not os.path.exists(ckpt):
        raise ValueError(f"Checkpoint {ckpt} does not exist")
    
    vae_ckpt = None
    model = load_model_from_config(config, ckpt, vae_ckpt)
    model.eval().cuda()

    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    image_blocks = gr.Blocks(css=css)
    with image_blocks as demo:
        gr.HTML(get_header())
        with gr.Group():
            with gr.Box():
                with gr.Row():

                    with gr.Column():
                        image = gr.Image(source='upload', tool=None, elem_id="image_upload", type="pil", label="Source Image")
                        instruction = gr.Textbox(lines=3, placeholder="Enter text to edit", label="Text")

                        cfg_text = gr.Slider(label="Guidance scale (TXT)", value=7.0, maximum=15,interactive=True)
                        cfg_image = gr.Slider(label="Guidance scale (IMG)", value=1.25, maximum=15,interactive=True)
                        
                        steps = gr.Slider(label="Steps", value=50, minimum=2, maximum=75, step=1,interactive=True)
                        resolution = gr.Slider(label="Resolution (long side)", value=512, minimum=256, maximum=768, step=64, interactive=True)

                        seed = gr.Slider(0, 10000, label='Seed', value=0, step=1)

                        with gr.Row(elem_id="prompt-container", mobile_collapse=False, equal_height=True):
                            btn = gr.Button(
                                "Edit!",
                                margin=False,
                                rounded=(False, True, True, False),
                                full_width=True,
                            )

                    # output
                    with gr.Column():
                        image_out = gr.Image(label="Output", elem_id="output-img", height=400, show_download_button=True)

                    partial_predict = partial(
                        predict, 
                        model, model_wrap, 
                        model_wrap_cfg,
                        null_token, # RESOLUTION
                    )

                    btn.click(
                        fn=partial_predict, 
                        inputs=[
                            resolution, image, instruction, seed, steps, cfg_text, cfg_image
                        ], 
                        outputs=[image_out])

                gr.HTML(
                    """
                        <div class="footer">
                            <p>
                            InstructDiffusion Demo
                            </p>
                        </div>
                        <div class="acknowledgments">
                                <p><h4>LICENSE</h4>
                        The model is licensed with a <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" style="text-decoration: underline;" target="_blank">CreativeML Open RAIL-M</a> license. The authors claim no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in this license. The license forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" target="_blank" style="text-decoration: underline;" target="_blank">read the license</a></p>
                    """
                )

    image_blocks.launch(share=True, max_threads=1).queue()


if __name__ == "__main__":
    main()