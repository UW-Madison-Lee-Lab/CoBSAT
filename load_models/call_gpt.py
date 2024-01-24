# seed: set

from environment import OPENAI_API_KEY
from openai import OpenAI
import os, base64, requests
from typing import Literal
from time import time

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def process_image(image_input, image_mode, image_input_detail):
    if image_mode == 'url':
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": image_input,
                "detail": image_input_detail,
            },
        }
    elif image_mode == 'path':
        image_content = {
            "type": "image",
            "image_url": {
                "url": f"data:image/jpg;base64,{encode_image(image_input)}",
                "detail": image_input_detail,
            },
        }
    else:
        raise ValueError("The image_mode must be either 'url' or 'path', not {image_mode}.")
    
    return image_content

def process_text(text_input):
    text_content = {
        "type": "text",
        "text": text_input,
    }
    
    return text_content

def prompt_image_eval(
    text_inputs, 
    image_inputs,
    image_mode,
    image_input_detail,
    instruction = "I will provide you with a few examples with text and images. Complete the example with the description of the next image. The description should be clear with main object, and include details such as color, texture, background, style, and action, if applicable. Tell me only the text prompt and I'll use your entire answer as a direct input to A Dalle-3. Never say other explanations. ",
    call_mode = 'micl', # micl or text only
):    
    contents = [process_text(instruction[0])]
    
    for i in range(len(text_inputs)):
        contents.append(process_text(text_inputs[i]))
        
        if call_mode == 'micl':
            if i < len(text_inputs) - 1:
                contents.append(process_image(image_inputs[i], image_mode, image_input_detail))
        
    contents.append(process_text(instruction[1]))
    messages = [{
        "role": "user",
        "content": contents,
    }]
    
    return messages
        

def call_gpt4v(
    text_inputs = ["Red", "Green", "Yellow"],
    image_mode: Literal['url', 'path'] = 'path',
    image_inputs = [
        "https://media.istockphoto.com/id/1189903200/photo/red-generic-sedan-car-isolated-on-white-background-3d-illustration.jpg?s=612x612&w=0&k=20&c=uRu3o_h5FVljLQHS9z0oyz-XjXzzXN_YkyGXwhdMrjs=",
        "https://media.istockphoto.com/id/186872128/photo/a-bright-green-hatchback-family-car.jpg?s=2048x2048&w=is&k=20&c=vy3UZdiZFG_lV0Mp_Nka2DC4CglOqEuujpC-ra5TWJ0="
    ],
    use_dalle = False,
    dalle_version: Literal['dall-e-2', 'dall-e-3'] = 'dall-e-3',
    image_input_detail: Literal['low', 'high'] = 'low',
    max_tokens = 300,
    image_output_size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] = "1024x1024",
    image_output_quality:  Literal['hd', 'standard'] = 'standard',
    seed = 123,
    instruction = [
        "I will provide you with a few examples with text and images. Complete the example with the description of the next image. The description should be clear with main object, and include details such as color, texture, background, style, and action, if applicable. Tell me only the text prompt and I'll use your entire answer as a direct input to A Dalle-3. Never say other explanations. ",
        '',
    ],
    call_mode = 'micl', # micl or text only
):
    
    if len(text_inputs) != (len(image_inputs)+1):
        raise ValueError("The number of text inputs must be equal to the number of image urls plus one.")
    if len(text_inputs) > 10:
        raise ValueError("The number of demonstrations must be less than or equal to 10.")

    output_dict = {}

    messages = prompt_image_eval(
        text_inputs, 
        image_inputs, 
        image_mode,
        image_input_detail,
        instruction,
        call_mode,
    )
    
    # Call GPT-4V to generate text description 
    
    payload = {
        'model':"gpt-4-vision-preview",
        'messages':messages,
        'max_tokens':max_tokens,
        'seed': seed,
    }
    
    gpt4v_start = time()
    if image_mode == 'url':
        response = client.chat.completions.create(**payload)
        output_dict['description'] = response.choices[0].message.content
    elif image_mode == 'path':
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload,
        )
        output_dict['description'] = response.json()['choices'][0]['message']['content']
    else:
        raise ValueError("The image_mode must be either 'url' or 'path', not {mode}.")    
    
    gpt4v_end = time()
    output_dict['gpt4v_time'] = gpt4v_end - gpt4v_start

    # Call DALL-E to generate image
    if use_dalle:
        dalle_start = time()
        response = client.images.generate(
            image_model=dalle_version,
            prompt=output_dict['description'],
            size=image_output_size,
            quality=image_output_quality,
            n=1,
        )

        dalle_end = time()
        output_dict['image_url'] = response.data[0].url
        output_dict['dalle_time'] = dalle_end - dalle_start
    
    return output_dict

def prompt_text_eval(
    prompt_idx,
    image_description,
    ground_truth_description,
):
    if prompt_idx == 1:
        request = f"Does the image description \"{image_description}\" describe {ground_truth_description}? Please answer yes or no without any other texts."
    elif prompt_idx == 2:
        request = f"Is it correct to say that \"{image_description}\" describes {ground_truth_description}? Please answer yes or no without any other texts."
    else:
        raise ValueError("The prompt must be either 1 or 2.")
    return request

def call_gpt3_chat(
    image_description,
    ground_truth_description,
    prompt_idx = 1, 
    seed = 123,
):
    request = prompt_text_eval(
        prompt_idx,
        image_description,
        ground_truth_description,
    )
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that only answer yes or no."},
            {"role": "user", "content": request},
        ],
        seed = seed,
    )
    return response.choices[0].message.content

def call_gpt3_completion(
    image_description,
    ground_truth_description,
    prompt_idx = 1, 
    seed = 123,
):

    request = prompt_text_eval(
        prompt_idx,
        image_description,
        ground_truth_description,
    )
    print(request)
    
    success = False
    retry = 0
    while not success:
        try:
            response = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=f"Q: {request} Please answer yes or no without any other texts. A: ",
                seed = seed,
            )
            success = True
        except KeyboardInterrupt:
            exit()
        except:
            retry += 1
            if retry >= 5:
                return "ERROR"
            
    return response.choices[0].text