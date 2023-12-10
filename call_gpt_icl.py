from openai import OpenAI
import os, base64, requests
from typing import Literal
from time import time
from environment import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def process_image(image_input, mode, image_input_detail):
    if mode == 'url':
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": image_input,
                "detail": image_input_detail,
            },
        }
    elif mode == 'path':
        image_content = {
            "type": "image",
            "image_url": {
                "url": f"data:image/jpg;base64,{encode_image(image_input)}",
                "detail": image_input_detail,
            },
        }
    else:
        raise ValueError("The mode must be either 'url' or 'path', not {mode}.")
    
    return image_content

def process_text(text_input):
    text_content = {
        "type": "text",
        "text": text_input,
    }
    
    return text_content

def get_prompt(
    text_inputs, 
    image_inputs,
    prompt_index, 
    mode,
    image_input_detail,
):    
    contents = []
    if prompt_index == 1:
        text_description = "I will provide you a few examples with text and image. Complete the example with the description of next image. Tell me only the text prompt and I'll use your entire answer as a direct input to A Dalle-3. Never say other explanations. "
        contents.append(process_text(text_description))
    elif prompt_index == 2:
        text_description = "I will provide you a few examples with text and image. Generate the clear description of the next image based on the pattern from previous examples. Your output will be directly used as input for DALL-E model."
        contents.append(process_text(text_description))
    
    for i in range(len(text_inputs)):
        contents.append(process_text(text_inputs[i]))
        
        if i < len(text_inputs) - 1:
            contents.append(process_image(image_inputs[i], mode, image_input_detail))
                    
    messages = [{
        "role": "user",
        "content": contents,
    }]
    
    return messages
        

def call_gpt(
    text_inputs = ["Red", "Green", "Yellow"],
    mode: Literal['url', 'path'] = 'path',
    image_inputs = [
        "https://media.istockphoto.com/id/1189903200/photo/red-generic-sedan-car-isolated-on-white-background-3d-illustration.jpg?s=612x612&w=0&k=20&c=uRu3o_h5FVljLQHS9z0oyz-XjXzzXN_YkyGXwhdMrjs=",
        "https://media.istockphoto.com/id/186872128/photo/a-bright-green-hatchback-family-car.jpg?s=2048x2048&w=is&k=20&c=vy3UZdiZFG_lV0Mp_Nka2DC4CglOqEuujpC-ra5TWJ0="
    ],
    use_dalle = False,
    dalle_version: Literal['dall-e-2', 'dall-e-3'] = 'dall-e-3',
    image_input_detail: Literal['low', 'high'] = 'low',
    prompt_index = 1,
    max_tokens = 300,
    image_output_size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] = "1024x1024",
    image_output_quality:  Literal['hd', 'standard'] = 'standard',
    seed = 123,
):
    
    if len(text_inputs) != (len(image_inputs)+1):
        raise ValueError("The number of text inputs must be equal to the number of image urls plus one.")
    if len(text_inputs) > 10:
        raise ValueError("The number of demonstrations must be less than or equal to 10.")

    output_dict = {}

    messages = get_prompt(
        text_inputs, 
        image_inputs, 
        prompt_index, 
        mode,
        image_input_detail,
    )
    
    # Call GPT-4V to generate text description 
    
    payload = {
        'model':"gpt-4-vision-preview",
        'messages':messages,
        'max_tokens':max_tokens,
        'seed': seed,
    }
    
    gpt4v_start = time()
    if mode == 'url':
        response = client.chat.completions.create(**payload)
        output_dict['description'] = response.choices[0].message.content
    elif mode == 'path':
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
        raise ValueError("The mode must be either 'url' or 'path', not {mode}.")    
    
    gpt4v_end = time()
    output_dict['gpt4v_time'] = gpt4v_end - gpt4v_start

    # Call DALL-E to generate image
    if use_dalle:
        dalle_start = time()
        response = client.images.generate(
            model=dalle_version,
            prompt=output_dict['description'],
            size=image_output_size,
            quality=image_output_quality,
            n=1,
        )

        dalle_end = time()
        output_dict['image_url'] = response.data[0].url
        output_dict['dalle_time'] = dalle_end - dalle_start
    
    return output_dict


import argparse
import random
import glob
import json


parser = argparse.ArgumentParser(description='seed_llama')
parser.add_argument('--shot', type=int, nargs='+', default=[1, 2, 4])
parser.add_argument('--misleading', type=bool, default=[False, True])
parser.add_argument('--max_file_count', type=int, default=1)

args = parser.parse_args()

max_file_count = args.max_file_count

dataset_1_list = [['black','blue','green','pink','purple','red','white','yellow'], ['one','two','three','four','five','six'], ['cartoon','cubism','oil','origami','sketch','watercolor'],['drink','eat','fly','run','sing','sit','sleep','wink'],['beach','classroom','forest','gym','library','office','park','street']]
space_1_list = ["color", "count", "style", "action", "background"]
dataset_2_list = [['bag','box','building','car','chair','cup','flower','leaf'],['apple','cat','chair','cup','dog','lion','person','shampoo'], ['apple','car','cat','chair','dog','flower','house','man'],['bird','cat','dog','lion','man','monkey','pig','woman'],['car','cat','chair','dog','man','monkey','robot','woman']]
space_2_list = ["object", "object", "object","animal", "object"]

task_type = ["odd", "even"]

for shot in args.shot:
    for misleading in args.misleading:
        base_path = "results/shot_" + str(shot) if misleading == False else "results/shot_" + str(shot) + "_m"
        for t, (dataset_1, space_1, dataset_2, space_2) in enumerate(zip(dataset_1_list, space_1_list, dataset_2_list, space_2_list)):
            for task in task_type:
                folder_path = base_path + "/task_" + str(2 * t + 1) + "/" if task == "odd" else base_path + "/task_" + str(2 * t + 2) + "/"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                x_list = dataset_1 if task == "odd" else dataset_2 
                theta_list = dataset_2 if task == "odd" else dataset_1

                while len(glob.glob(folder_path + '/*.json')) < max_file_count:
                    random.shuffle(x_list)
                    random.shuffle(theta_list)
                    x_m_list = x_list if misleading == False else [x + " " + theta for x, theta in zip(x_list, theta_list)]
                    theta = theta_list[shot+1]

                    text_inputs = []
                    image_inputs = []
                    save_path = folder_path + str(len(glob.glob(folder_path + '/*.jpg'))) + "_" + theta + "_"
                    print("========")
                    print(theta)
                    print("--------")
                    for i in range(shot+1):
                        image_path_i = "examples/" + space_1 + "_" + theta + "/" + x_list[i] + "_" + theta + ".jpg" if task == "odd" else "examples/" + space_1 + "_" + x_list[i] + "/" + theta + "_" + x_list[i] + ".jpg"                        
                        text_inputs.append(x_m_list[i] + ": ")
                        if i < shot:
                            image_inputs.append(image_path_i)

                        print(x_m_list[i])
                        save_path = save_path + "_" + x_list[i]
                    print("========")

                    out = call_gpt(text_inputs=text_inputs, image_inputs=image_inputs)
                    save_path = save_path + ".json"

                    with open(save_path,'w') as f:
                        json.dump(out, f)

