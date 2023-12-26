from openai import OpenAI
import os
from call_gpt import call_gpt4v
from environment import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()



import argparse
import random
import glob
import json


parser = argparse.ArgumentParser(description='gpt4v')
parser.add_argument('--shot', type=int, nargs='+', default=[1, 2, 4])
parser.add_argument('--misleading', type=bool, default=[False, True])
parser.add_argument('--max_file_count', type=int, default=1)

args = parser.parse_args()

#  data_folder = 'examples/'
data_folder = f"datasets"

max_file_count = args.max_file_count

dataset_1_list = [['black','blue','green','pink','purple','red','white','yellow']]
space_1_list = ["color"]
dataset_2_list = [['bag','box','building','car','chair','cup','flower','leaf']]
space_2_list = ["object"]

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
                        # image_path_i = f"{data_folder}/" + space_1 + "_" + theta + "/" + x_list[i] + "_" + theta + ".jpg" if task == "odd" else "examples/" + space_1 + "_" + x_list[i] + "/" + theta + "_" + x_list[i] + ".jpg"                        
                        image_path_i = f"{data_folder}/{space_1} {space_2}/{space_1} {theta}/{x_list[i]} {theta}.jpg" if task == "odd" else f"{data_folder}/{space_1} {space_2}/{space_1} {x_list[i]}/{theta} {x_list[i]}.jpg"
                        if not os.path.exists(image_path_i):
                            image_path_i = image_path_i.split('.jpg')[0] + '.webp'
                        text_inputs.append(f"{x_m_list[i]}: ")
                        if i < shot:
                            image_inputs.append(image_path_i)

                        print(x_m_list[i])
                        save_path = save_path + "_" + x_list[i]
                    print("========")

                    print(text_inputs)
                    print(image_inputs)
                    out = call_gpt4v(
                        text_inputs=text_inputs, 
                        image_inputs=image_inputs, 
                        mode = 'path', 
                        use_dalle = False
                    )
                    print(out)
                    save_path = save_path + ".json"

                    with open(save_path,'w') as f:
                        json.dump(out, f)

