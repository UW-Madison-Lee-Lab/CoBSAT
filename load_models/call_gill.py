# seed: set

import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'models/gill'))

from models.gill.gill import models
import torch
from helper import set_seed
from PIL import Image
from time import time
      
def load_gill(
    device = 'cuda',
    seed = 123,
):
    # Download the model checkpoint and embeddings to checkpoints/gill_opt/
    model_dir = f'{root_dir}/models/gill/checkpoints/gill_opt/'
    model = models.load_gill(model_dir, device = device)
    
    g_cuda = torch.Generator(device=device).manual_seed(seed)
    return model, g_cuda
    

def call_gill(
    model, 
    g_cuda,
    text_inputs = ["Red", "Green", "Yellow"],
    image_inputs = [
        "/data/yzeng58/micl/datasets/weather_pig/aurora_pig.jpg",
        "/data/yzeng58/micl/datasets/weather_pig/hailstorm_pig.jpg"
    ],
    seed = 123,
    gen_mode = 'text',
    instruction = [
        'You are a professional assistant and always answer my question directly and perfectly without any excuses.',
        'Based on the sequence, describe what the next image should be clearly, including details such as the main object, color, texture, background, action, style, if applicable. Your response should only contain a description of the image, and all other information can cause huge loss.',
    ],
    call_mode = 'micl', # 'micl' or 'text'
    history = None,
    save_history = False,
):
    set_seed(seed)
    
    prompt = []
    if history: prompt.extend(history)
    
    for i in range(len(text_inputs)):
        prompt.append(text_inputs[i])
        if call_mode == 'micl':
            if i < len(text_inputs) - 1:
                image = Image.open(image_inputs[i]).convert('RGB')
                prompt.append(image)
            
    if instruction[0]: prompt.insert(0, instruction[0])
    if instruction[1]: prompt.append(instruction[1])
            
    output_dict = {}
    if save_history: output_dict['history'] = prompt
    
    gill_start = time()
    
    if gen_mode == 'image':
        return_outputs = model.generate_for_images_and_texts(
            prompt, num_words=2, ret_scale_factor=100.0, generator=g_cuda)
        
        output_dict['description'] = return_outputs[0]
        
        if return_outputs[1]['decision'][0] == 'gen':
            output_dict['image'] = return_outputs[1]['gen'][0][0]
        else:
            output_dict['image'] = return_outputs[1]['ret'][0][0].resize((512, 512))
            print('Error!!!')
            
        if save_history: output_dict['history'] = output_dict['history'] + ' ' + output_dict['image']  
        
    elif gen_mode == 'text':
        output_dict['description'] = model.generate_for_images_and_texts(prompt, num_words=16, min_word_tokens=16)[0]
        if save_history: output_dict['history'] = output_dict['history'] + ' ' + output_dict['description']    
    
    gill_end = time()
    output_dict['time'] = gill_end - gill_start
    
    return output_dict
    
    
    
    