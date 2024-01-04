import os
root_dir = os.path.dirname(os.path.abspath(__file__))

from helper import set_seed, read_json, find_image
from configs import task_dataframe

def load_prompt(
    num_demos,
    misleading,
    task_id,
    x_idxs,
    theta_idxs,
    x_list,
    theta_list,
    instruction,
):
    text_inputs, image_inputs = [instruction], []
    
    for shot in range(num_demos+1):
        
        x = x_list[x_idxs[shot]]
        theta = theta_list[theta_idxs[shot]]
        
        if misleading:
            text_inputs.append(f"{x} {theta}")
        else:
            text_inputs.append(x)
            
        if shot < num_demos:
            image_inputs.append(find_image(
                root_dir, 
                task_id, 
                x_idxs[shot], 
                theta_idxs[shot], 
            ))
    
    return text_inputs, image_inputs
    

def load_dataset(
    num_demos,
    misleading,
    task_id, 
    num_prompt = 1000,
    seed = 123,
    instruction = "I will provide you a few examples with text and image. Complete the example with the description of next image. Tell me only the text prompt and I'll use your entire answer as a direct input to A Dalle-3. Never say other explanations. ",
):
    print("========"*3)
    print(f'Loading the dataset for task {task_id}...')
    print(f'| task type: {task_dataframe[task_id]["task_type"]}')
    print(f'| x_space: {task_dataframe[task_id]["x_space"]}')
    print(f'| theta_space: {task_dataframe[task_id]["theta_space"]}')
    print(f'| misleading: {misleading}')
    print(f'| num_demos: {num_demos}')
    
    set_seed(seed)
    
    prompts_list = read_json(f"{root_dir}/load_datasets/prompts_list.json")
    data_loader = []
    for i in range(num_prompt):
        item_inputs = prompts_list[i]
        text_inputs, image_inputs = load_prompt(
            num_demos,
            misleading,
            task_id,
            item_inputs["x_list"],
            item_inputs["theta_list"],
            task_dataframe[task_id]["x_list"],
            task_dataframe[task_id]["theta_list"],
            instruction,
        )
        data_loader.append({
            'x_list': 
            "text_inputs": text_inputs,
            "image_inputs": image_inputs,
        })
    print('Done!')
    print("========"*3)
    return data_loader
        
        
        
        