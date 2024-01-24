import os
root_dir = os.path.dirname(os.path.abspath(__file__))

from helper import set_seed, read_json, find_image
from configs import task_dataframe

def load_prompt(
    shot,
    prompt_type,
    task_id,
    x_idxs,
    theta_idxs,
    x_list,
    theta_list,
):
    text_inputs, image_inputs = [], []
    theta = theta_list[theta_idxs[shot+1]]
    x_demos = []
    
    for demo_idx in range(shot+1):
        x_idx = x_list[x_idxs[demo_idx]]
        x_demos.append(x_idx)
        theta_idx = theta_list[theta_idxs[demo_idx]]
        
        if prompt_type == 1:
            # misleading
            text_inputs.append(f"{x_idx} {theta_idx}")
        else:
            text_inputs.append(x_idx)
            
        if demo_idx < shot:
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
        'theta': theta_list[theta_idxs[shot+1]],
    }
    

def load_dataset(
    shot,
    prompt_type,
    task_id, 
    num_prompt = 1000,
    seed = 123,
):
    print("========"*3)
    print(f'Loading the dataset for task {task_id}...')
    print(f'| task type: {task_dataframe[task_id]["task_type"]}')
    print(f'| x_space: {task_dataframe[task_id]["x_space"]}')
    print(f'| theta_space: {task_dataframe[task_id]["theta_space"]}')
    print(f'| prompt_type: {prompt_type}')
    print(f'| shot: {shot}')
    
    set_seed(seed)
    
    prompts_list = read_json(f"{root_dir}/load_datasets/prompts_list.json")
    data_loader = []
    for i in range(num_prompt):
        item_inputs = prompts_list[i]
        input_dict = load_prompt(
            shot,
            prompt_type,
            task_id,
            item_inputs["x_list"],
            item_inputs["theta_list"],
            task_dataframe[task_id]["x_list"],
            task_dataframe[task_id]["theta_list"],
        )
        input_dict['save_path'] = f"{i}_{input_dict['theta']}_{'_'.join(input_dict['x_list'])}"
        data_loader.append(input_dict)
    print('Done!')
    print("========"*3)
    return data_loader
        
        
        
        