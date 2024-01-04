# seed:set

import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from helper import set_seed

from PIL import Image 
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from time import time

def load_emu2(
    devices = None, 
    device_mem = {0:'30GiB',1:'30GiB',2:'30GiB'},
    model_path =  '/data/yzeng58/.cache/huggingface/hub/models--BAAI--Emu2/snapshots/fa835ec101e52da5e081695107e1ddd3c7c4d88a',
):
    if devices: os.environ["CUDA_VISIBLE_DEVICES"] = devices
    
    tokenizer = AutoTokenizer.from_pretrained("BAAI/Emu2") # "BAAI/Emu2-Chat"

    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            "BAAI/Emu2", # "BAAI/Emu2-Chat"
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True)  

    device_map = infer_auto_device_map(model, max_memory=device_mem, no_split_module_classes=['Block','LlamaDecoderLayer'])  
    # input and output logits should be on same device
    device_map["model.decoder.lm.lm_head"] = 0

    model = load_checkpoint_and_dispatch(
        model, 
        model_path,
        device_map=device_map).eval()
    
    return model, tokenizer

def call_emu2(
    model, 
    tokenizer,
    text_inputs = ["Brown", "White", 'Black'],
    image_inputs = [
        f'{root_dir}/models/Emu/Emu2/examples/dog2.jpg',
        f'{root_dir}/models/Emu/Emu2/examples/dog3.jpg',
    ],
    seed = 123,
    instruction = "I will provide you a few examples with text and image. Complete the example with the description of next image. Tell me only the text prompt and I'll use your entire answer as a direct input to A Dalle-3. Never say other explanations. "
):
    set_seed(seed)
    
    prompt = instruction
    for i in range(len(text_inputs)):
        prompt = prompt + text_inputs[i]
        if i < len(text_inputs) - 1:
            prompt = prompt + "[<IMG_PLH>]"
            
    images = [Image.open(image_inputs[i]).convert('RGB') for i in range(len(image_inputs))]
    
    inputs = model.build_input_ids(
        text=[prompt],
        tokenizer=tokenizer,
        image=images,
    )
    
    output_dict = {}
    emu2_start = time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image=inputs["image"].to(torch.bfloat16),
            max_new_tokens=64,
            length_penalty=-1,
            temperature=0,
        )
    emu2_end = time()
    output_dict['description'] = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    output_dict['time'] = emu2_end - emu2_start

    return output_dict
