import os, argparse
from load_model import load_model
root_dir = os.path.dirname(os.path.abspath(__file__))

from helper import save_json, read_json, set_seed
from load_dataset_ft import load_dataset
from configs import task_dataframe, item_dict, item2word

import pandas as pd


def image_description(
    task_id,
    data,
):
    csv = pd.read_csv(task_dataframe[task_id]["task_type"] + ".csv", sep = ",")
    find = csv[csv["image"] == os.path.basename(data["image_inputs"][-1])] 

    return find.values[0][1]


if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='gen_dataset_ft')
    parser.add_argument('--shot', type=int, nargs='+', default=2)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--task_ids', type=int, nargs='+', default=list(task_dataframe.keys()))
    parser.add_argument('--train_num', type=int, nargs='+', default=5)

    args = parser.parse_args()
    
    args_dict = vars(args)
    print("########"*3)
    print('## Experiment Setting:')
    print("########"*3)
    for key, value in args_dict.items():
        print(f"| {key}: {value}")
    
    set_seed(args.seed)

    dataset_ft = []

    for task_id in args.task_ids:

        data_loader = load_dataset(
            args.shot,
            task_id,
            args.train_num,
        )

        instruction = [
            'You are a professional assistant and always answer my question directly and perfectly without any excuses.',
            "\nBased on the sequence, describe what the next image should be clearly, including details such as the main object, color, texture, background, action, style, if applicable. Your response should only contain a description of the image, and all other information can cause huge loss.",
        ]
        
        for idx, data in enumerate(data_loader):

            prompt = ""
            prompt = prompt + instruction[0] + "\n"
            for i in range(len(data["text_inputs"])):
                prompt = prompt + data["text_inputs"][i]
                if i < len(data["text_inputs"]) - 1:
                    prompt = prompt + " Picture " + str(i+1) + ": " + "<img>" + data["image_inputs"][i] + "</img>" + "\n"
            prompt = prompt + instruction[1]


            data_ft = {
                "id": str(task_id) + "_" + str(idx),
                "conversations": [
                {
                    "from": "user",
                    "value": prompt
                },
                {
                    "from": "assistant",
                    "value": image_description(task_id, data)
                }
                ]
            }

            dataset_ft.append(data_ft)
    
    save_json(dataset_ft, '/pvc/ceph-block-kangwj1995/wisconsin/copy2/Qwen-VL/dataset_ft.json')



