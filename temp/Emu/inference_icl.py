import argparse

import json

import torch
from models.modeling_emu import Emu
from utils import process_img, process_video

import random
import glob
import os
import numpy as np

image_placeholder = "[IMG]" + "<image>" * 32 + "[/IMG]"
image_system_msg = "You will be presented with an image: [IMG]ImageContent[/IMG]. You will be able to see the image after I provide it to you. Please answer my questions based on the given image."
video_system_msg = "You are a helpful assistant and you will be presented with a video consisting of multiple chronological images: [IMG]ImageContent[/IMG]. You will be able to see the video after I provide it to you. Please answer my questions based on the given video."

icl_system_msg = "I will provide you a few examples with text and image. Complete the example with the description of next image. Tell me only the text prompt and I'll use your entire answer as a direct input to A Dalle-3. Never say other explanations. "

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instruct",
        action='store_true',
        default=False,
        help="Load Emu-I",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default='',
        help="Emu ckpt path",
    )

    parser.add_argument('--shot', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--misleading', type=bool, nargs='+', default=[False, True])
    parser.add_argument('--max_file_count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=123)


    args = parser.parse_args()

    return args


def prepare_model(model_name, args):
    with open(f'models/{model_name}.json', "r", encoding="utf8") as f:
        model_cfg = json.load(f)
    print(f"=====> model_cfg: {model_cfg}")

    model = Emu(**model_cfg, cast_dtype=torch.float, args=args)

    if args.instruct:
        print('Patching LoRA...')
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.decoder.lm = get_peft_model(model.decoder.lm, lora_config)

    print(f"=====> loading from ckpt_path {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    if 'module' in ckpt:
        ckpt = ckpt['module']
    msg = model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f"=====> get model.load_state_dict msg: {msg}")

    return model


def Emu_inference(image_list, text_sequence, system='', instruct=True, max_new_tokens=128, beam_size=5, length_penalty=0.0):
    if instruct:
        prompt = f"{system} [USER]: {text_sequence} [ASSISTANT]:".strip()
    else:
        prompt = text_sequence

    print(f"===> prompt: {prompt}")

    samples = {"image": torch.cat(image_list, dim=0), "prompt": prompt}

    output_text = emu_model.generate(
        samples,
        max_new_tokens=max_new_tokens,
        num_beams=beam_size,
        length_penalty=length_penalty,
        repetition_penalty=1.0,
    )[0].strip()

    print(f"===> output: {output_text}\n")

    return output_text


def pretrain_example():
    # prepare in-context learning example
    image_text_sequence = [
        

        process_img(img_path='examples/dog.png', device=args.device),
        #'Hello, what is your name?',
        'There are two dogs.',
        process_img(img_path='examples/panda.png', device=args.device),
        'There are three pandas.',
        process_img(img_path='examples/sunflower.png', device=args.device),
    
    ]
    interleaved_sequence_1 = ''
    image_list_1 = []
    for item in image_text_sequence:
        if isinstance(item, str):  # text
            interleaved_sequence_1 += item
        else:  # image
            image_list_1.append(item)
            interleaved_sequence_1 += image_placeholder

    # Pretrained Model Inference
    # -- in-context learning
    Emu_inference(image_list_1, interleaved_sequence_1, instruct=False)


if __name__ == '__main__':

    args = parse_args()

    # initialize and load model
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    emu_model = prepare_model('Emu-14B', args)
    emu_model.to(args.device).to(torch.bfloat16)


    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


    max_file_count = args.max_file_count

    dataset_1_list = [['black','blue','green','pink','purple','red','white','yellow'], ['one','two','three','four','five','six'], ['cartoon','cubism','oil','origami','sketch','watercolor'],['drink','eat','fly','run','sing','sit','sleep','wink'],['beach','classroom','forest','gym','library','office','park','street']]
    space_1_list = ["color", "count", "style", "action", "background"]
    dataset_2_list = [['bag','box','building','car','chair','cup','flower','leaf'],['apple','cat','chair','cup','dog','lion','person','shampoo'], ['apple','car','cat','chair','dog','flower','house','man'],['bird','cat','dog','lion','man','monkey','pig','woman'],['car','cat','chair','dog','man','monkey','robot','woman']]
    space_2_list = ["object", "object", "object","animal", "object"]

    task_type = ["odd", "even"]

    for shot in args.shot:
        for misleading in args.misleading:
            base_path = "results_prompt_2/shot_" + str(shot) if misleading == False else "results_prompt_2/shot_" + str(shot) + "_m"
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

                        input_tokens = []
                        #input_tokens.append(text_description)
                        save_path = folder_path + str(len(glob.glob(folder_path + '/*.json'))) + "_" + theta + "_"
                        print("========")
                        print(theta)
                        print("--------")
                        for i in range(shot+1):
                            image_path_i = "examples/" + space_1 + "_" + theta + "/" + x_list[i] + "_" + theta + ".jpg" if task == "odd" else "examples/" + space_1 + "_" + x_list[i] + "/" + theta + "_" + x_list[i] + ".jpg"
                            image_i = process_img(img_path=image_path_i, device=args.device)

                            input_tokens.append(x_m_list[i] + ": ")
                            if i < shot:
                                input_tokens.append(image_i)

                            print(x_m_list[i])
                            save_path = save_path + "_" + x_list[i]
                        print("========")

                        save_path = save_path + ".json"

                        image_text_sequence = input_tokens
                        
                        interleaved_sequence_1 = ''
                        image_list_1 = []
                        for item in image_text_sequence:
                            if isinstance(item, str):  # text
                                interleaved_sequence_1 += item
                            else:  # image
                                image_list_1.append(item)
                                interleaved_sequence_1 += image_placeholder

                        return_outputs = Emu_inference(image_list_1, interleaved_sequence_1, instruct=True, system=icl_system_msg)

                        with open(save_path,'w') as f:
                            json.dump(return_outputs, f)



