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
from environment import EMU2_PATH, EMU2_GEN_PATH
from diffusers import DiffusionPipeline

def load_emu2(
    device = ['cuda:0', 'cuda:1', 'cuda:2'],
    gen_mode = 'text',
):  
    if gen_mode == 'text':
        
        tokenizer = AutoTokenizer.from_pretrained("BAAI/Emu2") # "BAAI/Emu2-Chat"

        if isinstance(device, str): # single-gpu
            model = AutoModelForCausalLM.from_pretrained(
                "BAAI/Emu2", # "BAAI/Emu2-Chat"
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True).to(device).eval()
        elif isinstance(device, list):  # multi-gpu 
            with init_empty_weights():
                model = AutoModelForCausalLM.from_pretrained(
                    "BAAI/Emu2", # "BAAI/Emu2-Chat"
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True)  
                
            device = {int(i[5:]):'30GiB' for i in device}

            device_map = infer_auto_device_map(model, max_memory=device, no_split_module_classes=['Block','LlamaDecoderLayer'])  
            # input and output logits should be on same device
            device_map["model.decoder.lm.lm_head"] = 0

            model = load_checkpoint_and_dispatch(
                model, 
                EMU2_PATH,
                device_map=device_map).eval()
        
        return model, tokenizer
    
    elif gen_mode == 'image':
        if isinstance(device, list): # multi-gpu
            raise NotImplementedError(f'Emu2: gen_mode {gen_mode} not implemented for multi-gpu cases.')
            
        try:
            # For the non-first time of using, you can init the pipeline directly
            pipe = DiffusionPipeline.from_pretrained(
                EMU2_GEN_PATH,
                custom_pipeline="pipeline_emu2_gen",
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                variant="bf16",
            )
        except:
            # For the first time of using,
            # you need to download the huggingface repo "BAAI/Emu2-GEN" to local first
            multimodal_encoder = AutoModelForCausalLM.from_pretrained(
                f"{EMU2_GEN_PATH}/multimodal_encoder",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                variant="bf16"
            )
            tokenizer = AutoTokenizer.from_pretrained(f"{EMU2_GEN_PATH}/tokenizer")
            
            pipe = DiffusionPipeline.from_pretrained(
                EMU2_GEN_PATH,
                custom_pipeline="pipeline_emu2_gen",
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                variant="bf16",
                multimodal_encoder=multimodal_encoder,
                tokenizer=tokenizer,
            )
            
        pipe.to(device)
        return pipe, None
    else:
        raise NotImplementedError(f'gen_mode {gen_mode} not implemented.')

def call_emu2(
    model, 
    tokenizer,
    text_inputs = ["Brown", "White", 'Black'],
    image_inputs = [
        f'{root_dir}/models/Emu/Emu2/examples/dog2.jpg',
        f'{root_dir}/models/Emu/Emu2/examples/dog3.jpg',
    ],
    seed = 123,
    gen_mode = 'text',
    instruction = "Based on the sequence, describe the next image clearly, including details such as the main object, color, texture, weather, action, style, if applicable. ",
):
    set_seed(seed)
    
    if gen_mode == 'text':
        prompt = ''
        for i in range(len(text_inputs)):
            prompt = prompt + text_inputs[i]
            if i < len(text_inputs) - 1:
                prompt = prompt + "[<IMG_PLH>]"
        prompt = prompt + instruction        
        
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
    elif gen_mode == 'image':
        prompt = []
        for i in range(len(text_inputs)):
            prompt.append(text_inputs[i])
            if i < len(text_inputs) - 1:
                prompt.append(Image.open(image_inputs[i]).convert('RGB'))
        prompt.append(instruction)
        
        output_dict = {}
        
        emu2_start = time()
        outputs = model(prompt)  
        emu2_end = time()
    
        output_dict['image'] = outputs.image
        # output_dict['description'] = outputs.text # not sure
        output_dict['time'] = emu2_end - emu2_start
        
    else:
        raise NotImplementedError(f'gen_mode {gen_mode} not implemented.')

    return output_dict
