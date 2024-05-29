import os, sys
root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)
from environment import CLAUDE_API_KEY  

import anthropic, base64
from time import time
from PIL import Image
    
def ensure_jpeg(image_path):
    try:
        with Image.open(image_path) as img:
            if img.format != 'JPEG':
                img = img.convert('RGB')  # Convert to RGB if not
                
                new_image_path = image_path.replace('.jpg', '.jpeg').replace('datasets', 'archive/datasets_jpeg')
                new_image_dir = os.path.dirname(new_image_path)
                os.makedirs(new_image_dir, exist_ok = True)
                
                img.save(new_image_path, 'JPEG')  # Save as JPEG
                image_path = new_image_path
    except IOError:
        print("Unable to open the image. Ensure the file is a valid image.")
        
    return image_path

# Function to encode the image
def encode_image(image_path):
    image_path = ensure_jpeg(image_path)
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def process_image(image_input):
    image_content = {
        'type': 'image',
        'source': {
            'type': 'base64',
            'media_type': 'image/jpeg',
            'data': encode_image(image_input),
        }
    }
    return image_content

def process_text(text_input):
    if text_input: 
        text_content = {
            "type": "text",
            "text": text_input,
        }
    else:
        text_content = None
    return text_content

def prompt_image_eval(
    text_inputs, 
    image_inputs,
    instruction = [
        "I will provide you with a few examples with text and images. Complete the example with the description of the next image. The description should be clear with main object, and include details such as color, texture, background, style, and action, if applicable. Tell me only the text prompt and I'll use your entire answer as a direct input to A Dalle-3. Never say other explanations. ", 
        '',
    ],
    call_mode = 'micl', # micl or text only
):    
    text = process_text(instruction[0])
    contents = [text] if text else []
    
    for i in range(len(text_inputs)):
        contents.append(process_text(text_inputs[i]))
        
        if call_mode == 'micl':
            if i < len(text_inputs) - 1:
                contents.append(process_image(image_inputs[i]))
        
    text = process_text(instruction[1])
    if text: contents.append(text)
    messages = [{
        "role": "user",
        "content": contents,
    }]
    
    return messages

def load_claude(api_key):
    client = anthropic.Anthropic(
        api_key = CLAUDE_API_KEY[api_key]
    )
    return client

def call_claude(
    client,
    text_inputs = ["Red", "Green", "Yellow"],
    image_inputs = [
        f'{root_dir}/datasets/color_box/red_box.jpg',
        f'{root_dir}/datasets/color_box/green_box.jpg',
    ],
    instruction = [
        "I will provide you a few examples with text and image. Complete the example with the description of next image. Never say other explanations. ",
        'Give me the description of the next image.',
    ],
    call_mode = 'micl',
    max_tokens = 1024,
    history = None,
    save_history = False,
):
    messages = prompt_image_eval(
        text_inputs,
        image_inputs, 
        instruction,
        call_mode,
    )
    if history is not None: messages = history + messages
    
    output_dict = {}
    if save_history: output_dict = {'history': messages}
    
    claude_start = time()
    response = client.messages.create(
        model = "claude-3-haiku-20240307",
        max_tokens = max_tokens,
        messages = messages,
        temperature = 0,
    ).content[0].text
    output_dict['description'] = response  
    claude_end = time()
    output_dict['claude_time'] = claude_end - claude_start
    
    if save_history:
        output_dict['history'] = messages + [{
            'role': 'assistant',
            'content': output_dict['description'],
        }]
    
    return output_dict