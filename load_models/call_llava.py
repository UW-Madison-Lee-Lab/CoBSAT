import torch, os, json, argparse, sys, random, re
import numpy as np
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.eval.run_llava import eval_model
from llava.utils import disable_torch_init
from time import time

root_dir = os.path.dirname(os.getcwd())

def load_llava(device = 'cuda'):
    # Configure LlaVA
    model_path = f"{root_dir}/evaluation/llava-v1.5-13b/"

    model_name = "llava-v1.5-13b"
    tokenizer, llava_model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )
    print('LlaVA model loaded.')
    llava_args = type('Args', (), {
        "conv_mode": None,
        "sep": ",",
        # "temperature": 0.2,
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    disable_torch_init()

    return tokenizer, llava_model, image_processor, context_len, llava_args

def call_llava(
    tokenizer,
    llava_model,
    image_processor,
    context_len,
    llava_args,
    text_inputs = ["Red", "Green", "Yellow"],
    image_inputs = [
        "https://media.istockphoto.com/id/1189903200/photo/red-generic-sedan-car-isolated-on-white-background-3d-illustration.jpg?s=612x612&w=0&k=20&c=uRu3o_h5FVljLQHS9z0oyz-XjXzzXN_YkyGXwhdMrjs=",
        "https://media.istockphoto.com/id/186872128/photo/a-bright-green-hatchback-family-car.jpg?s=2048x2048&w=is&k=20&c=vy3UZdiZFG_lV0Mp_Nka2DC4CglOqEuujpC-ra5TWJ0="
    ],
    seed = 123,
):

    set_seed(seed)
    
    instruction = "I will provide you a few examples with text and image. Complete the example with the description of next image. Tell me only the text prompt and I'll use your entire answer as a direct input to A Dalle-3. Never say other explanations. "
    prompt = instruction
    for i in range(len(text_inputs)):
        prompt = prompt + text_inputs[i]
        if i < len(text_inputs) - 1:
            prompt = prompt + "<image-placeholder>"

    print(prompt)
    output_dict = {}
    llava_start = time()
    output_dict['description'] = eval_model(prompt, image_inputs, tokenizer, llava_model, image_processor, context_len, llava_args)
    llava_end = time()
    output_dict['time'] = llava_end - llava_start

    return output_dict


"""
tokenizer, llava_model, image_processor, context_len, llava_args = load_llava(device = 'cuda')
out= call_llava(tokenizer, llava_model, image_processor, context_len, llava_args)
print(out)
"""