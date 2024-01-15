import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import argparse
from helper import set_seed, read_json
from configs import task_dataframe, supported_models
from load_dataset import load_dataset
from tqdm import tqdm


def remove_error(
    task_id,
    shot,
    misleading,
    model,
    seed,
    max_file_count = 1000,
    eval_mode = 'text',
):
    task_type = task_dataframe[task_id]['task_type']
    category_space = {}
    category_space['detail'], category_space['obj'] = task_type.split('_')
    
    misleading_flag = "_m" if misleading else ""
    set_seed(seed)
    
    data_loader = load_dataset(
        shot,
        misleading,
        task_id,
        max_file_count,
    )
    
    base_path = f"{root_dir}/results/exps/{model}_{eval_mode}/shot_{shot}{misleading_flag}"
    folder_path = f"{base_path}/task_{task_id}"
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return
    
    for count in tqdm(range(max_file_count), desc = f"Cleaning {model}_{eval_mode}/shot_{shot}{misleading_flag}/task_{task_id}"):
            
        input_dict = data_loader[count]
        input_dict['x'] = input_dict['x_list'][-1]
        
        file_path = f"{folder_path}/{input_dict['save_path']}.json"
        if os.path.exists(file_path):
            output_dict = read_json(file_path)
    
            if output_dict['description'].lower() == 'error':
                print(file_path)
                # delete this file
                os.remove(file_path)
    

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description = 'Evaluate the results using LLaVA.')
    parser.add_argument('--model', type = str, default = 'qwen', choices = supported_models, help = 'model')
    parser.add_argument('--task_id', type = int, nargs = '+', default = list(task_dataframe.keys()), help = 'task id')
    parser.add_argument('--shot', type = int, nargs = '+', default = [2,4,6,8], help = 'shot')
    parser.add_argument('--misleading', type = int, nargs = '+', default = [0,1], help = 'misleading', choices = [0,1])
    parser.add_argument('--device', type = str, default = 'cuda', help = 'device')
    parser.add_argument('--seed', type = int, default = 123, help = 'seed')
    parser.add_argument('--eval_mode', type = str, default = 'text', help = 'evaluation mode', choices = ['text', 'image'])
    parser.add_argument('--max_file_count', type = int, default = 1000, help = 'max file count')
    
    args = parser.parse_args()
    
    # print experiment configuration
    args_dict = vars(args)
    print("########"*3)
    print('## Experiment Setting:')
    print("########"*3)
    for key, value in args_dict.items():
        print(f"| {key}: {value}")
    
    
    for task_id in args.task_id:
        for shot in args.shot:
            for misleading in args.misleading:
                remove_error(
                    task_id,
                    shot,
                    misleading,
                    args.model,
                    args.seed,
                    max_file_count = args.max_file_count,
                    eval_mode = args.eval_mode,
                )