import os
root_dir = os.path.dirname(os.path.abspath(__file__))

from helper import set_seed, read_json, find_image
from configs import task_dataframe

from itertools import permutations, combinations

def load_prompt(
    shot,
    task_id,
    x_idxs,
    theta_idxs,
    x_list,
    theta_list,
):
    text_inputs, image_inputs = [], []
    #theta = theta_list[theta_idxs[shot+1]]
    theta = theta_list[theta_idxs[0]]
    x_demos = []
    
    for demo_idx in range(shot+1):
        x_idx = x_list[x_idxs[demo_idx]]
        x_demos.append(x_idx)
        
        text_inputs.append(x_idx)
            
        if demo_idx < shot + 1: ##########
            image_inputs.append(find_image(
                root_dir, 
                task_id, 
                x_idxs[demo_idx], 
                theta, 
            ))
    
    return {
        "text_inputs": text_inputs,
        "image_inputs": image_inputs,
        "x_list": x_demos,
        'theta': theta,
    }
    

def load_dataset(
    shot,
    task_id, 
    train_num,
    seed = 123,
):
    set_seed(seed)
    
    data_loader = []

    index_list = list(range(train_num))
    x_lists = list(permutations(index_list, shot + 1))
    theta_lists = list(permutations(index_list, 1))

    print("========"*3)
    print(f'Loading the dataset for task {task_id}...')
    print(f'| task type: {task_dataframe[task_id]["task_type"]}')
    print(f'| x_space: {task_dataframe[task_id]["x_space"]}')
    print(f'| theta_space: {task_dataframe[task_id]["theta_space"]}')
    print(f'| shot: {shot}')

    for x_list in x_lists:
        for theta_list in theta_lists:
            input_dict = load_prompt(
                shot,
                task_id,
                x_list,
                theta_list,
                task_dataframe[task_id]["x_list"],
                task_dataframe[task_id]["theta_list"],
            )
            data_loader.append(input_dict)

    print('Done!')
    print("========"*3)
    return data_loader
        
        
        
        
