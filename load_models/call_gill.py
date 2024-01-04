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
    instruction = "I will provide you a few examples with text and image. Complete the example with the description of next image. Tell me only the text prompt and I'll use your entire answer as a direct input to A Dalle-3. Never say other explanations. ",
):
    set_seed(seed)
    
    prompt = [instruction]
    for i in range(len(text_inputs)):
        prompt.append(text_inputs[i])
        if i < len(text_inputs) - 1:
            image = Image.open(image_inputs[i]).convert('RGB')
            prompt.append(image)
            
    output_dict = {}
    gill_start = time()
    if gen_mode == 'image':
        output_dict['description'] = model.generate_for_images_and_texts(
            prompt, num_words=2, ret_scale_factor=100.0, generator=g_cuda)
    elif gen_mode == 'text':
        output_dict['description'] = model.generate_for_images_and_texts(prompt, num_words=16, min_word_tokens=16)
    gill_end = time()
    output_dict['time'] = gill_end - gill_start
    
    return output_dict
    
    
    
    