# seed -- set

import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from helper import set_seed 
from time import time

def load_qwen(device = 'cuda'):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map=device, trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    return model, tokenizer

    
def call_qwen(
    model, 
    tokenizer,
    text_inputs = ["Red", "Green", "Yellow"],
    image_inputs = [
        "https://media.istockphoto.com/id/1189903200/photo/red-generic-sedan-car-isolated-on-white-background-3d-illustration.jpg?s=612x612&w=0&k=20&c=uRu3o_h5FVljLQHS9z0oyz-XjXzzXN_YkyGXwhdMrjs=",
        "https://media.istockphoto.com/id/186872128/photo/a-bright-green-hatchback-family-car.jpg?s=2048x2048&w=is&k=20&c=vy3UZdiZFG_lV0Mp_Nka2DC4CglOqEuujpC-ra5TWJ0="
    ],
    seed = 123,
    instruction = [
        'You are a professional assistant and always answer my question directly and perfectly without any excuses.',
        "\nBased on the sequence, describe what the next image should be clearly, including details such as the main object, color, texture, background, action, style, if applicable. Your response should only contain a description of the image, and all other information can cause huge loss.",
    ],
    call_mode = 'micl', # 'micl' or 'text'
    history = None,
    save_history = False,
):
    set_seed(seed)
    
    # get prompt
    messages = [{'text': instruction[0]}]
    for i in range(len(text_inputs)):
        messages.append({'text': text_inputs[i]})
        if call_mode == 'micl': 
            if i < len(text_inputs) - 1:
                messages.append({'image': image_inputs[i]})
    messages.append({'text': instruction[1]})
    
    output_dict = {}
    qwen_start = time()
    query = tokenizer.from_list_format(messages)
    output_dict['description'], output_dict['history'] = model.chat(tokenizer, query=query, history=history)
    qwen_end = time()
    output_dict['time'] = qwen_end - qwen_start
    
    if not save_history: output_dict.pop('history')
    
    return output_dict
    
