import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from helper import set_seed 

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
):
    set_seed(seed)
    
    # get prompt
    instruction = "I will provide you a few examples with text and image. Complete the example with the description of next image. Tell me only the text prompt and I'll use your entire answer as a direct input to A Dalle-3. Never say other explanations. "
    messages = [{'text': instruction}]
    for i in range(len(text_inputs)):
        messages.append({'text': text_inputs[i]})
        if i < len(text_inputs) - 1:
            messages.append({'image': image_inputs[i]})
    
    query = tokenizer.from_list_format(messages)
    response, history = model.chat(tokenizer, query=query, history=None)
    return response
    
