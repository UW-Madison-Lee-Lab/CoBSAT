import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
from tqdm import notebook

from gill import models
from gill import utils

# Download the model checkpoint and embeddings to checkpoints/gill_opt/
model_dir = 'checkpoints/gill_opt/'
model = models.load_gill(model_dir)
model.sd_pipe.safety_checker = None

g_cuda = torch.Generator(device='cuda').manual_seed(1337)

def get_result(prompt, name):
  return_outputs = model.generate_for_images_and_texts(
      prompt, num_words=2, ret_scale_factor=100.0, generator=g_cuda)

  if return_outputs[1]['decision'][0] == 'gen':
      im = return_outputs[1]['gen'][0][0]
      im.save(name)
  else:
      im = return_outputs[1]['ret'][0][0].resize((512, 512))
      im.save(name)
      print('Error!!!')


def get_image(name):
  extensions = ['jpg', 'webp', 'jpeg', 'png']
  found_image = None
  for ext in extensions:
      try:
          image_path = name+f'.{ext}'
          found_image = Image.open(image_path).convert('RGB')
          break
      except FileNotFoundError:
          continue

  if found_image is None:
      print(f"No valid image found for {name} !")
  return found_image



import argparse
import random
import glob


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

                    get_result(input_tokens, save_path)

