import hydra

import pyrootutils
import os
import torch

from omegaconf import OmegaConf
import json
from typing import Optional
import transformers
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

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


def decode_image_text(generate_ids, tokenizer, save_path=None):

    boi_list = torch.where(generate_ids == tokenizer(
        BOI_TOKEN, add_special_tokens=False).input_ids[0])[0]
    eoi_list = torch.where(generate_ids == tokenizer(
        EOI_TOKEN, add_special_tokens=False).input_ids[0])[0]

    if len(boi_list) == 0 and len(eoi_list) == 0:
        text_ids = generate_ids
        texts = tokenizer.decode(text_ids, skip_special_tokens=True)
        #print(texts)

    else:
        boi_index = boi_list[0]
        eoi_index = eoi_list[0]

        text_ids = generate_ids[:boi_index]
        if len(text_ids) != 0:
            texts = tokenizer.decode(text_ids, skip_special_tokens=True)
            #print(texts)

        image_ids = (generate_ids[boi_index+1:eoi_index] -
                     image_id_shift).reshape(1, -1)

        images = tokenizer.decode_image(image_ids)

        images[0].save(save_path)


device = "cuda"

tokenizer_cfg_path = 'configs/tokenizer/seed_llama_tokenizer_hf.yaml'
tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
tokenizer = hydra.utils.instantiate(
    tokenizer_cfg, device=device, load_diffusion=True)

transform_cfg_path = 'configs/transform/clip_transform.yaml'
transform_cfg = OmegaConf.load(transform_cfg_path)
transform = hydra.utils.instantiate(transform_cfg)

model_cfg = OmegaConf.load('configs/llm/seed_llama_14b.yaml')
model = hydra.utils.instantiate(model_cfg, torch_dtype=torch.float16)
model = model.eval().to(device)

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

#############################

import random
import glob
import argparse

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

                while len(glob.glob(folder_path + '/*.jpg')) < max_file_count:
                    random.shuffle(x_list)
                    random.shuffle(theta_list)
                    x_m_list = x_list if misleading == False else [x + " " + theta for x, theta in zip(x_list, theta_list)]
                    theta = theta_list[shot+1]

                    input_tokens = tokenizer.bos_token  + s_token
                    save_path = folder_path + theta + "_"
                    print("========")
                    print(theta)
                    print("--------")
                    for i in range(shot+1):
                        image_path_i = "examples/" + space_1 + "_" + theta + "/" + x_list[i] + "_" + theta + ".jpg" if task == "odd" else "examples/" + space_1 + "_" + x_list[i] + "/" + theta + "_" + x_list[i] + ".jpg"
                        image_i = Image.open(image_path_i).convert('RGB')
                        image_tensor_i = transform(image_i).to(device)
                        img_ids_i = tokenizer.encode_image(image_torch=image_tensor_i)
                        img_ids_i = img_ids_i.view(-1).cpu().numpy()
                        img_tokens_i = BOI_TOKEN + ''.join([IMG_TOKEN.format(item)
                                                        for item in img_ids_i]) + EOI_TOKEN
                        input_tokens = input_tokens + x_m_list[i] + ": " if i == shot else input_tokens + x_m_list[i] +  ": " + img_tokens_i
                        print(x_m_list[i])
                        save_path = save_path + "_" + x_list[i]
                    print("========")
                    input_tokens = input_tokens + e_token + sep
                    save_path = save_path + ".jpg"
                    generate_ids = generate(tokenizer, input_tokens, generation_config, model)        
                    decode_image_text(generate_ids, tokenizer, save_path)
