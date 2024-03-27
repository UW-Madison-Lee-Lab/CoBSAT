import os, sys, json
root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)
from environment import GEMINI_API_KEY

import google.generativeai as genai
import PIL.Image
from time import time
from helper import retry_if_fail

def load_gemini(prompt_type, api_key):
    genai.configure(api_key = GEMINI_API_KEY[api_key])
    if prompt_type == 'caption':
        model = genai.GenerativeModel('gemini-pro')
    else:
        model = genai.GenerativeModel('gemini-pro-vision')
    return model
    
@retry_if_fail
def call_gemini(
    model,
    text_inputs = ["Red", "Green", "Yellow"],
    image_inputs = [
        f'{root_dir}/datasets/color_box/red_box.jpg',
        f'{root_dir}/datasets/color_box/green_box.jpg',
    ],
    instruction = [
        "I will provide you a few examples with text and image. Complete the example with the description of next image. Tell me only the text prompt and I'll use your entire answer as a direct input to A Dalle-3. Never say other explanations. ",
        '',
    ],
    call_mode = 'micl',
    history = None,
    save_history = False
):
    prompt = [instruction[0]]
    
    for i in range(len(text_inputs)):
        prompt.append(text_inputs[i])
        if call_mode == 'micl':
            if i < len(text_inputs) - 1:
                image = PIL.Image.open(image_inputs[i])
                prompt.append(image)
    prompt.append(instruction[1])
    
    output_dict = {}
    gemini_start = time()
    # if (history is not None) or save_history:
    #     Multiturn chat is not enabled for models/gemini-pro-vision'
    #     if history is None: history = []
    #     chat = model.start_chat(history = history)
    #     response = chat.send_message(
    #         prompt, 
    #         stream = False,
    #         generation_config=genai.types.GenerationConfig(temperature = 0),
    #     )
    #     output_dict['history'] = chat.history
        
    if history is None: history = []
    response = model.generate_content(
        history + prompt, 
        stream = False,
        generation_config=genai.types.GenerationConfig(temperature = 0),
    )

    if call_mode == 'micl': response.resolve()
    gemini_end = time()
    output_dict['time'] = gemini_end - gemini_start
    output_dict['description'] = response.text
    if save_history: output_dict['history'] = prompt + [response.text]
    
    return output_dict