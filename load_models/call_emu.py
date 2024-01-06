# seed: set

import os, sys, json
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from environment import EMU_IMAGE_PATH, EMU_TEXT_PATH
sys.path.append(os.path.join(root_dir, 'models/Emu/Emu1'))

from models.modeling_emu import Emu
from models.pipeline import EmuGenerationPipeline

from utils import process_img


import torch
from helper import set_seed
from PIL import Image
from time import time

image_placeholder = "[IMG]" + "<image>" * 32 + "[/IMG]"

def Emu_inference(emu_model, image_list, text_sequence, system='', instruct=True, max_new_tokens=128, beam_size=5, length_penalty=0.0):
    if instruct:
        prompt = f"{system} [USER]: {text_sequence} [ASSISTANT]:".strip()
    else:
        prompt = text_sequence

    print(f"===> prompt: {prompt}")

    samples = {"image": torch.cat(image_list, dim=0), "prompt": prompt}

    output_text = emu_model.generate(
        samples,
        max_new_tokens=max_new_tokens,
        num_beams=beam_size,
        length_penalty=length_penalty,
        repetition_penalty=1.0,
    )[0].strip()

    print(f"===> output: {output_text}\n")

    return output_text



def load_emu(
    device = 'cuda',
    seed = 123,
    gen_mode = 'image',
):
    if gen_mode == 'text':
        args = type('Args', (), {
            "instruct": True,
            "ckpt_path": EMU_TEXT_PATH, # "/pvc/ceph-block-kangwj1995/wisconsin/Emu/Emu/Emu-instruct.pt", ####
            "device": torch.device(device),
        })()

        with open(f'{root_dir}/models/Emu/Emu1/models/Emu-14B.json', "r", encoding="utf8") as f:
            model_cfg = json.load(f)
        print(f"=====> model_cfg: {model_cfg}")

        model = Emu(**model_cfg, cast_dtype=torch.float, args=args)

        if args.instruct:
            print('Patching LoRA...')
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model.decoder.lm = get_peft_model(model.decoder.lm, lora_config)

        print(f"=====> loading from ckpt_path {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        if 'module' in ckpt:
            ckpt = ckpt['module']
        msg = model.load_state_dict(ckpt, strict=False)
        model.eval()
        model.to(args.device).to(torch.bfloat16)
        print(f"=====> get model.load_state_dict msg: {msg}")

        return model

    elif gen_mode == 'image':
        args = type('Args', (), {
            "instruct": False,
            "ckpt_path": EMU_IMAGE_PATH, # "/pvc/ceph-block-kangwj1995/wisconsin/Emu/Emu/pretrain", ####
            "device": torch.device(device),
        })()

        model = EmuGenerationPipeline.from_pretrained(
            path=args.ckpt_path,
            args=args,
        )
        model = model.bfloat16().cuda()

        return model

def call_emu(
    emu_model, 
    text_inputs = ['Yellow', 'White', 'Black'],
    image_inputs = [
        f'{root_dir}/models/Emu/Emu2/examples/dog2.jpg',
        f'{root_dir}/models/Emu/Emu2/examples/dog3.jpg',
    ],
    seed = 123,
    gen_mode = 'text',
    device = 'cuda',
    instruction = "I will provide you a few examples with text and image. Complete the example with the description of next image. Tell me only the text prompt and I'll use your entire answer as a direct input to A Dalle-3. Never say other explanations. ",
):
    set_seed(seed)
    
    if gen_mode == 'image':

        prompt = []
        for i in range(len(text_inputs)):
            prompt.append(text_inputs[i])
            if i < len(text_inputs) - 1:
                image = Image.open(image_inputs[i]).convert('RGB')
                prompt.append(image)

        output_dict = {}
        emu_start = time()

        image = emu_model(
            prompt,
            height=512,
            width=512,
            guidance_scale=10.,
        )
        output_dict['description'] = "null"
        output_dict['image'] = image

        emu_end = time()
        output_dict['time'] = emu_end - emu_start

        return output_dict

    elif gen_mode == 'text':

        prompt = []
        for i in range(len(text_inputs)):
            prompt.append(text_inputs[i])
            if i < len(text_inputs) - 1:
                image = process_img(img_path=image_inputs[i],device=device)
                prompt.append(image)
                
        interleaved_sequence = ''
        image_list = []
        for item in prompt:
            if isinstance(item, str):  # text
                interleaved_sequence += item
            else:  # image
                image_list.append(item)
                interleaved_sequence += image_placeholder

        output_dict = {}
        emu_start = time()

        output_dict['description'] = Emu_inference(emu_model, image_list, interleaved_sequence, instruct=True, system=instruction)
        emu_end = time()
        output_dict['time'] = emu_end - emu_start

        return output_dict
    
    
    
    