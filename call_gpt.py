from environment import OPENAI_API_KEY
from openai import OpenAI
import os, base64, requests

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def call_gpt(
    text_inputs = ["Red", "Green", "Yellow"],
    mode = 'url', # 'url' or 'path'
    image_inputs = [
        "https://media.istockphoto.com/id/1189903200/photo/red-generic-sedan-car-isolated-on-white-background-3d-illustration.jpg?s=612x612&w=0&k=20&c=uRu3o_h5FVljLQHS9z0oyz-XjXzzXN_YkyGXwhdMrjs=",
        "https://media.istockphoto.com/id/186872128/photo/a-bright-green-hatchback-family-car.jpg?s=2048x2048&w=is&k=20&c=vy3UZdiZFG_lV0Mp_Nka2DC4CglOqEuujpC-ra5TWJ0="
    ],
    max_tokens = 300,
    image_size = "1024x1024",
    quality = 'standard',
):
    
    if len(text_inputs) != (len(image_inputs)+1):
        raise ValueError("The number of text inputs must be equal to the number of image urls plus one.")
    if len(text_inputs) > 10:
        raise ValueError("The number of demonstrations must be less than or equal to 10.")
    
    word_dict = {
        0: 'first',
        1: 'second',
        2: 'third',
        3: 'fourth',
        4: 'fifth',
        5: 'sixth',
        6: 'seventh',
        7: 'eighth',
        8: 'ninth',
        9: 'tenth',
    }

    # Design the prompt
    text_description = f"I want you to generate the image based on the given examples. Tell me only the text prompt to use it as an input of Dalle3. Never say other explanations. Tell me only the prompt! "
    for i, text_input in enumerate(text_inputs):
        if i < len(text_inputs) - 1:
            text_description += f"{text_input}: (see the {word_dict[i]} image)."
        else:
            text_description += f"{text_input}:"
    
    text_content = [{
        "type": "text",
        "text": text_description,
    }]
    
    image_content = []
    for image_input in image_inputs:
        if mode == 'url':
            image_content.append({
                "type": "image_url",
                "image_url": {
                "url": image_input,
                },
            })
        elif mode == 'path':
            image_content.append({
                "type": "image",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(image_input)}",
                },
            })
        else:
            raise ValueError("The mode must be either 'url' or 'path', not {mode}.")
        
        messages = [{
            "role": "user",
            "content": text_content + image_content,
        }]
    
    # Call GPT-4V to generate text description 
    payload = {
        'model':"gpt-4-vision-preview",
        'messages':messages,
        'max_tokens':max_tokens,
    }
    
    if mode == 'url':
        response = client.chat.completions.create(**payload)
        description = response.choices[0].message.content
    elif mode == 'path':
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload,
        )
        description = response.json()['choices'][0]['message']['content']
    
    # Call DALL-E to generate image
    response = client.images.generate(
        model="dall-e-3",
        prompt=description,
        size=image_size,
        quality=quality,
        n=1,
    )

    image_url = response.data[0].url
    
    return image_url





