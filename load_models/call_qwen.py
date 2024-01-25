# Update qwen/finetune.py

import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from helper import set_seed, find_caption, find_image, get_ft_model_dir
from time import time
from load_dataset import get_prompt
from peft import AutoPeftModelForCausalLM, PeftModel

def load_qwen(
    device = 'cuda',
    finetuned = False,
    shot = 2, 
    gen_mode = 'text',
    prompt_type = 'default',
):  
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    
    if finetuned:
        ft_path = get_ft_model_dir(
            'qwen',
            gen_mode,
            shot,
            prompt_type,
        )
        # model = AutoPeftModelForCausalLM.from_pretrained(
        #     ft_path, # path to the output directory
        #     device_map="auto",
        #     trust_remote_code=True,
        # ).eval()
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map=device, trust_remote_code=True).eval()
        model = PeftModel.from_pretrained(model, ft_path)
    else: 
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map=device, trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        return model, tokenizer

def load_qwen_prompt(
    instruction,
    text_inputs,
    call_mode,
    image_inputs,
):  
    # get prompt
    messages = [{'text': instruction[0]}]
    for i in range(len(text_inputs)):
        messages.append({'text': text_inputs[i]})
        if call_mode == 'micl': 
            if i < len(text_inputs) - 1:
                messages.append({'image': image_inputs[i]})
    messages.append({'text': instruction[1]})
    return messages
    
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
    
    messages = load_qwen_prompt(
        instruction,
        text_inputs,
        call_mode,
        image_inputs,
    )
    
    output_dict = {}
    qwen_start = time()
    query = tokenizer.from_list_format(messages)
    output_dict['description'], output_dict['history'] = model.chat(
        tokenizer, 
        query=query, 
        history=history,
        system = "You are a professional assistant and always answer my question directly and perfectly without any excuses. If I ask you to predict, you should always give me the prediction with the highest probability or even random guess.",
    )
    qwen_end = time()
    output_dict['time'] = qwen_end - qwen_start
    
    if not save_history: output_dict.pop('history')
    
    return output_dict

def load_ft_qwen_prompt(
    task_id,
    data_idx,
    prompt_type,
    text_inputs,
    image_inputs,
    x_idx,
    theta,
):
    query = get_prompt(
        text_inputs,
        image_inputs,
        prompt_type,
        task_id,
        model = 'qwen',
        gen_mode = 'text',
        history = None,
    )
    
    messages = load_qwen_prompt(
        query['instruction'],
        query['text_inputs'],
        'micl',
        query['image_inputs'],
    )
    
    prompt, image_count = '', 0
    for message in messages:
        if 'text' in message:
            prompt += message['text']
        elif 'image' in message:
            prompt = f"{prompt} Picture {image_count+1}: <img>{image_inputs[image_count]}</img>\n"
            image_count += 1
        else:
            raise ValueError(f"Unknown message type: {message.keys()}")
        
    
    prompt_ft = {
        'id': f"{task_id}_{data_idx}",
        "conversations":[
            {
                "from": "user",
                "value": prompt,
            },
            {
                "from": "assistant",
                "value": find_caption(find_image(
                    root_dir,
                    task_id,
                    x_idx,
                    theta,
                )),
            }
        ]
    }
    return prompt_ft

def ft_qwen(
    data_path,
    output_dir,
    use_lora = True,
    q_lora = False,
    lora_r = 64,
    lora_alpha = 16,
    lora_dropout = 0.05,
    seed = 123,
):
    set_seed(seed)
    from models.QwenVL.finetune import train, ModelArguments, DataArguments, TrainingArguments, LoraArguments
    model_args = ModelArguments(
        model_name_or_path="Qwen/Qwen-VL-Chat"
    )
    data_args = DataArguments(
        data_path=data_path, 
        eval_data_path=None, 
        lazy_preprocess=True,
    )
    training_args = TrainingArguments(
        cache_dir = None,
        optim = "adamw_torch",
        model_max_length = 2048,
        use_lora = use_lora,
        output_dir = output_dir,
        bf16 = True,
        fix_vit = True,
        num_train_epochs = 5,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = 8,
        evaluation_strategy = "no",
        save_strategy = "steps",
        save_steps = 1000,
        save_total_limit = 10,
        learning_rate = 1e-5,
        weight_decay=0.1,
        adam_beta2=0.95,
        warmup_ratio = 0.01,
        lr_scheduler_type = 'cosine',
        logging_steps = 1,
        report_to = 'none',
        gradient_checkpointing = True,
    )
    lora_args = LoraArguments(
        lora_r = lora_r,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        q_lora = q_lora
    )
    train(
        model_args,
        data_args,
        training_args,
        lora_args,
    )
    