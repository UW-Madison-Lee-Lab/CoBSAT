# seed: set

import os, sys, hydra
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'models/SEED'))
from environment import SEED_PROJECT_ROOT

import torch
from helper import set_seed
from PIL import Image
from time import time
from configs import instruction_dict
from torch.utils.data import Dataset

from omegaconf import OmegaConf

image_placeholder = "[IMG]" + "<image>" * 32 + "[/IMG]"

generation_config = {
    'temperature': 1.0,
    'num_beams': 1,
    'max_new_tokens': 512,
    'top_p': 0.5,
    'do_sample': True
}

s_token = "[INST] "
e_token = " [/INST]"
sep = "\n"

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

IMG_FLAG = '<image>'
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
image_id_shift = 32000

def generate(tokenizer, input_tokens, generation_config, model):

    input_ids = tokenizer(
        input_tokens, add_special_tokens=False, return_tensors='pt').input_ids
    input_ids = input_ids.to("cuda")
    generate_ids = model.generate(
        input_ids=input_ids,
        **generation_config
    )
    generate_ids = generate_ids[0][input_ids.shape[1]:]

    return generate_ids

def decode_image_text(generate_ids, tokenizer, gen_mode):
    # start of image
    boi_list = torch.where(generate_ids == tokenizer(
        BOI_TOKEN, add_special_tokens=False).input_ids[0])[0]
    # end of image
    eoi_list = torch.where(generate_ids == tokenizer(
        EOI_TOKEN, add_special_tokens=False).input_ids[0])[0]

    if len(boi_list) == 0 and len(eoi_list) == 0:
        text_ids = generate_ids
        texts = tokenizer.decode(text_ids, skip_special_tokens=True)
        images = None

    else:
        boi_index = boi_list[0]
        eoi_index = eoi_list[0]
        text_ids = generate_ids[:boi_index]
        if len(text_ids) != 0:
            texts = tokenizer.decode(text_ids, skip_special_tokens=True)
        else:
            texts = "null"
        image_ids = (generate_ids[boi_index+1:eoi_index] -
                    image_id_shift).reshape(1, -1)
        images = tokenizer.decode_image(image_ids)
        images = images[0]          

    if gen_mode == "text":
        return texts
    elif gen_mode == "image":
        return texts, images


def load_seed(
    device = 'cuda',
    seed = 123,
):
    set_seed(seed)
    os.environ["PROJECT_ROOT"] = SEED_PROJECT_ROOT
    
    tokenizer_cfg_path = f'{root_dir}/models/SEED/configs/tokenizer/seed_llama_tokenizer_hf.yaml'
    tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
    tokenizer = hydra.utils.instantiate(
        tokenizer_cfg, device=device, load_diffusion=True)

    transform_cfg_path = f'{root_dir}/models/SEED/configs/transform/clip_transform.yaml'
    transform_cfg = OmegaConf.load(transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    model_cfg = OmegaConf.load(f'{root_dir}/models/SEED/configs/llm/seed_llama_14b.yaml')
    model = hydra.utils.instantiate(model_cfg, torch_dtype=torch.float16)
    model = model.eval().to(device)

    return model, tokenizer, transform

def preprocess(
    query,
    tokenizer,
    instruction,
    history,
    call_mode,
    transform,
    device,
    output_mode = 'eval_sample', # ['eval_sample', 'train_sample']
    max_len = 2048,
):
    text_inputs, image_inputs = query['text_inputs'], query['image_inputs']
    
    input_tokens = tokenizer.bos_token  + s_token + instruction[0]
    if history is not None: input_tokens += history.replace(e_token, sep)
    
    if output_mode == 'eval_sample':
        num_images = len(text_inputs) - 1 
    elif output_mode == 'train_sample':
        num_images = len(text_inputs)
    else:
        raise ValueError(f'output_mode {output_mode} not supported')
    
    for i in range(len(text_inputs)):
        
        input_tokens = input_tokens + text_inputs[i]
        if call_mode == 'micl':
            if i < num_images:
                image = Image.open(image_inputs[i]).convert('RGB')
                image_tensor = transform(image).to(device)
                img_ids = tokenizer.encode_image(image_torch=image_tensor)
                img_ids = img_ids.view(-1).cpu().numpy()
                img_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(item)
                                                for item in img_ids]) + EOI_TOKEN
                input_tokens = input_tokens + img_tokens

    input_tokens += instruction[1]
    
    if output_mode == 'eval_sample':
        return input_tokens + e_token + sep
    elif output_mode == 'train_sample': 
        input_ids, output_ids = [], []
        output_ids += tokenizer(
            input_tokens + e_token + sep, 
            add_special_tokens=False, 
            return_tensors='pt',
        ).input_ids.squeeze()
        
        seps = input_tokens.split(BOI_TOKEN)
        input_tokens = BOI_TOKEN.join(seps[:-1]) + e_token + sep
        
        input_ids += tokenizer(
            input_tokens, 
            add_special_tokens=False, 
            return_tensors='pt',
        ).input_ids.squeeze()
        
        input_ids += [tokenizer.pad_token_id] * (max_len - len(input_ids))
        output_ids += [IGNORE_TOKEN_ID] * (max_len - len(output_ids))
        input_ids = torch.tensor(input_ids, dtype=torch.int)
        output_ids = torch.tensor(output_ids, dtype=torch.int)
        
        return dict(
            input_ids=input_ids,
            labels=output_ids,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )
    else:
        raise ValueError(f'output_mode {output_mode} not supported')

def call_seed(
    model,
    tokenizer,
    transform,
    text_inputs = ["Red", "Green", "Yellow"],
    image_inputs = [
        f"{root_dir}/datasets/weather_pig/aurora_pig.jpg",
        f"{root_dir}/datasets/weather_pig/hailstorm_pig.jpg"
    ],
    seed = 123,
    gen_mode = 'text',
    device = 'cuda',
    instruction = [
        "I will provide you a few examples with text and image. Complete the example with the description of next image. Tell me only the text prompt and I'll use your entire answer as a direct input to A Dalle-3. Never say other explanations. ",
        '',
    ],
    call_mode = 'micl', # 'micl' or 'text' 
    history = None,
    save_history = False,
):
    set_seed(seed)
    
    input_tokens = preprocess(
        query = {'text_inputs': text_inputs, 'image_inputs': image_inputs},
        tokenizer = tokenizer,
        instruction = instruction,
        history = history,
        call_mode = call_mode,
        transform = transform,
        device = device,
        output_mode = 'eval_sample',
    )

    output_dict = {}
    if save_history: output_dict = {'history': input_tokens}
    
    seed_start = time()
    if gen_mode == 'image':
        generate_ids = generate(tokenizer, input_tokens, generation_config, model)
        output_dict['description'], output_dict['image'] = decode_image_text(generate_ids, tokenizer, gen_mode)
        if save_history: output_dict['history'] += ' ' + output_dict['description']
        seed_end = time()
        output_dict['time'] = seed_end - seed_start

    elif gen_mode == 'text':
        generate_ids = generate(tokenizer, input_tokens, generation_config, model)
        output_dict['description'] = decode_image_text(generate_ids, tokenizer, gen_mode)
        if save_history: output_dict['history'] += ' ' + output_dict['description']
        seed_end = time()
        output_dict['time'] = seed_end - seed_start
    else:
        raise ValueError(f'gen_mode {gen_mode} not supported')

    return output_dict

# fine-tune
import transformers
from transformers import TrainingArguments, Trainer
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from peft import LoraConfig, get_peft_model
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    fix_vit: bool = True


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "attn.c_proj", "w1", "w2"] ##["in_proj","out_proj","c_fc"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    
class LoadData(Dataset):
    def __init__(
        self, 
        data,
        tokenizer,
        transform,
        gen_mode = 'image',
        history = None,
        call_mode = 'micl',
    ):
        self.data = data
        self.gen_mode = gen_mode
        self.tokenizer = tokenizer
        self.history = history
        self.call_mode = call_mode
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instruction = instruction_dict['default'][self.gen_mode]['seed']
        return preprocess(
            self.data[idx], 
            self.tokenizer, 
            instruction, 
            self.history, 
            self.call_mode, 
            self.transform, 
            'cuda',
            'train_sample',
        )

def ft_seed(
    data_ft,
    output_dir,
    gen_mode = 'image',
    history = None,
    call_mode = 'micl',
    use_lora = True,
    lora_r = 64,
    lora_alpha = 16,
    lora_dropout = 0.05,
    seed = 123,
):
    set_seed(seed)
    model, tokenizer, transform = load_seed()
    
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
    
    lora_config = LoraConfig(
        r = lora_r,
        lora_alpha = lora_alpha,
        target_modules = [
            "self_attn.q_proj", 
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ],
        lora_dropout = lora_dropout,
        bias = "none",
        task_type = 'CAUSAL_LM',
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    data_ft = LoadData(
        data = data_ft,
        tokenizer = tokenizer,
        transform = transform,
        gen_mode = gen_mode,
        history = history,
        call_mode = call_mode,
    )
    
    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        train_dataset = data_ft,
    )

    trainer.train()
    trainer.save_state()
    
    # Save the final model
    model_save_path = output_dir
    trainer.save_model(model_save_path)
    