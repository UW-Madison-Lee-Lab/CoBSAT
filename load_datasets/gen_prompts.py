import argparse, random, os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from helper import save_json
from configs import data_modes
from copy import deepcopy
from itertools import permutations

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Generate prompts')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max_file_count', type=int, default=1000)
    parser.add_argument('--list_length', type=int, default=10)
    parser.add_argument('--data_modes', type=str, nargs='+', default=data_modes, choices=data_modes)

    args = parser.parse_args()

    seed = args.seed
    max_file_count = args.max_file_count
    list_length = args.list_length
    data_modes = args.data_modes

    for data_mode in data_modes:
        random.seed(seed)

        if data_mode in ['inference', 'ft_test']:
            if data_mode == 'inference':
                x_list = list(range(list_length))
                theta_list = list(range(list_length))
            else:
                index_list = list(range(list_length//2, list_length))
                x_list = deepcopy(index_list)
                theta_list = deepcopy(index_list)
                
            prompts_list = []

            for i in range(max_file_count):
                random.shuffle(x_list)
                random.shuffle(theta_list)
                prompt_dict = {}
                prompt_dict["x_list"] = x_list.copy()
                prompt_dict["theta_list"] = theta_list.copy()
                prompts_list.append(prompt_dict)
        elif data_mode == 'ft_train':
            prompts_list = list(range(list_length//2))
        else:
            raise NotImplementedError(f"Unknown data_mode: {data_mode}!")
            
        save_json(prompts_list, f"{root_dir}/load_datasets/prompts_list_{data_mode}.json")
        

