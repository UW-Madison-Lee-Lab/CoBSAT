# seed: set

import os, sys, json, hydra
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'models/SEED'))

import torch
from helper import set_seed
from PIL import Image
from time import time

from omegaconf import OmegaConf
from typing import Optional
import transformers
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


import glob
import numpy as np

from transformers import set_seed

image_placeholder = "[IMG]" + "<image>" * 32 + "[/IMG]"

generation_config = {
    'temperature': 1.0,
    'num_beams': 1,
    'max_new_tokens': 512,
    'top_p': 0.5,
    'do_sample': True
}

s_token = "[INST] "
e_token = " [/INST]"
sep = "\n"

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

IMG_FLAG = '<image>'
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
image_id_shift = 32000

def generate(tokenizer, input_tokens, generation_config, model):

    input_ids = tokenizer(
        input_tokens, add_special_tokens=False, return_tensors='pt').input_ids
    input_ids = input_ids.to("cuda")
    generate_ids = model.generate(
        input_ids=input_ids,
        **generation_config
    )
    generate_ids = generate_ids[0][input_ids.shape[1]:]

    return generate_ids

def decode_image_text(generate_ids, tokenizer, gen_mode):

    boi_list = torch.where(generate_ids == tokenizer(
        BOI_TOKEN, add_special_tokens=False).input_ids[0])[0]
    eoi_list = torch.where(generate_ids == tokenizer(
        EOI_TOKEN, add_special_tokens=False).input_ids[0])[0]

    if len(boi_list) == 0 and len(eoi_list) == 0:
        text_ids = generate_ids
        texts = tokenizer.decode(text_ids, skip_special_tokens=True)
        images = None

    else:
        try:
            boi_index = boi_list[0]
            eoi_index = eoi_list[0]
            text_ids = generate_ids[:boi_index]
            if len(text_ids) != 0:
                texts = tokenizer.decode(text_ids, skip_special_tokens=True)
            else:
                texts = "null"
            image_ids = (generate_ids[boi_index+1:eoi_index] -
                        image_id_shift).reshape(1, -1)
            images = tokenizer.decode_image(image_ids)
            images = images[0]
        except:
            texts = "error"    
            images = None             

    if gen_mode == "text":
        return texts
    elif gen_mode == "image":
        return texts, images


def load_seed(
    device = 'cuda',
    seed = 123,
):

    tokenizer_cfg_path = 'models/SEED/configs/tokenizer/seed_llama_tokenizer_hf.yaml'
    tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
    tokenizer = hydra.utils.instantiate(
        tokenizer_cfg, device=device, load_diffusion=True)

    transform_cfg_path = 'models/SEED/configs/transform/clip_transform.yaml'
    transform_cfg = OmegaConf.load(transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    model_cfg = OmegaConf.load('models/SEED/configs/llm/seed_llama_14b.yaml')
    model = hydra.utils.instantiate(model_cfg, torch_dtype=torch.float16)
    model = model.eval().to(device)

    return model, tokenizer, transform

def call_seed(
    model,
    tokenizer,
    transform,
    text_inputs = ["Red", "Green", "Yellow"],
    image_inputs = [
        "/data/yzeng58/micl/datasets/weather_pig/aurora_pig.jpg",
        "/data/yzeng58/micl/datasets/weather_pig/hailstorm_pig.jpg"
    ],
    seed = 123,
    gen_mode = 'text',
    device = 'cuda',
    instruction = "I will provide you a few examples with text and image. Complete the example with the description of next image. Tell me only the text prompt and I'll use your entire answer as a direct input to A Dalle-3. Never say other explanations. ",
):
    set_seed(seed)
    
    if gen_mode == 'text':
        input_tokens = tokenizer.bos_token  + s_token + instruction
    else:
        input_tokens = tokenizer.bos_token  + s_token
    
    for i in range(len(text_inputs)):

        input_tokens = input_tokens + text_inputs[i] + ": "
        if i < len(text_inputs) - 1:
            image = Image.open(image_inputs[i]).convert('RGB')
            image_tensor = transform(image).to(device)
            img_ids = tokenizer.encode_image(image_torch=image_tensor)
            img_ids = img_ids.view(-1).cpu().numpy()
            img_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(item)
                                            for item in img_ids]) + EOI_TOKEN
            input_tokens = input_tokens + img_tokens

    input_tokens = input_tokens + e_token + sep

    output_dict = {}
    seed_start = time()
    if gen_mode == 'image':
        generate_ids = generate(tokenizer, input_tokens, generation_config, model)
        output_dict['description'], img = decode_image_text(generate_ids, tokenizer, gen_mode)
        seed_end = time()
        output_dict['time'] = seed_end - seed_start
        
        return output_dict, img

    elif gen_mode == 'text':
        generate_ids = generate(tokenizer, input_tokens, generation_config, model)
        output_dict['description'] = decode_image_text(generate_ids, tokenizer, gen_mode)
        seed_end = time()
        output_dict['time'] = seed_end - seed_start
        
        return output_dict

    
    
    
    