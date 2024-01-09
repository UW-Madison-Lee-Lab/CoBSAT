# seed: set

import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from helper import set_seed
from models.llava.llava.model.builder import load_pretrained_model
from models.llava.llava.eval.run_llava import eval_model
from models.llava.llava.mm_utils import get_model_name_from_path
from models.llava.llava.utils import disable_torch_init
from time import time


def load_llava(device = 'cuda'):
    # Configure LlaVA
    # model_path = f"{root_dir}/models/llava-v1.5-13b/"
    model_path = "liuhaotian/llava-v1.5-13b"

    tokenizer, llava_model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device = device,
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
    device = 'cuda',
):

    set_seed(seed)
    
    # prompt = "I will provide you with a few examples with text and images. Complete the example with the description of the next image. Tell me only the text prompt and I'll use your entire answer as a direct input to A Dalle-3. Never say other explanations. "
    prompt = ''
    for i in range(len(text_inputs)):
        prompt = prompt + text_inputs[i]
        if i < len(text_inputs) - 1:
            prompt = prompt + "<image-placeholder>"
    prompt = prompt + "\nBased on the sequence, describe the next image to be generated clearly, including details such as the main object, color, texture, background, action, style, if applicable. "

    output_dict = {}
    llava_start = time()
    output_dict['description'] = eval_model(
        prompt, 
        image_inputs, 
        tokenizer, 
        llava_model, 
        image_processor, 
        context_len, 
        llava_args, 
        device = device,
    )

    llava_end = time()
    output_dict['time'] = llava_end - llava_start

    return output_dict
