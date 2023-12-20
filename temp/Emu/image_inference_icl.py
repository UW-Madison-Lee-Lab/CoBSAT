# -*- coding: utf-8 -*-

import argparse

from PIL import Image
from models.pipeline import EmuGenerationPipeline


import random
import glob
import os

import torch
import numpy as np



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
        help="Emu Decoder ckpt path",
    )

    parser.add_argument('--shot', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--misleading', type=bool, nargs='+', default=[False, True])
    parser.add_argument('--max_file_count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=123)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    # NOTE
    # Emu Decoder Pipeline only supports pretrain model
    # Using instruct tuning model as image encoder may cause unpredicted results
    assert args.instruct is False, "Image Generation currently do not support instruct tuning model"

    pipeline = EmuGenerationPipeline.from_pretrained(
        path=args.ckpt_path,
        args=args,
    )
    pipeline = pipeline.bfloat16().cuda()



    # set seed for all experiments
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

                        input_tokens = []
                        save_path = folder_path + str(len(glob.glob(folder_path + '/*.jpg'))) + "_" + theta + "_"
                        print("========")
                        print(theta)
                        print("--------")
                        for i in range(shot+1):
                            image_path_i = "examples/" + space_1 + "_" + theta + "/" + x_list[i] + "_" + theta + ".jpg" if task == "odd" else "examples/" + space_1 + "_" + x_list[i] + "/" + theta + "_" + x_list[i] + ".jpg"
                            image_i = Image.open(image_path_i).convert('RGB')

                            input_tokens.append(x_m_list[i] + ": ")
                            if i < shot:
                                input_tokens.append(image_i)

                            print(x_m_list[i])
                            save_path = save_path + "_" + x_list[i]
                        print("========")

                        save_path = save_path + ".jpg"

                        image = pipeline(
                            input_tokens,
                            height=512,
                            width=512,
                            guidance_scale=10.,
                        )

                        image.save(save_path)


