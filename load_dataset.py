import os
root_dir = os.path.dirname(os.path.abspath(__file__))

from helper import set_seed, read_json, find_image, find_caption
from configs import task_dataframe, instruction_dict, num_prompt_dict
from itertools import permutations

def load_inputs(
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
        
        if prompt_type == 'misleading':
            # misleading
            theta_idx = theta_list[theta_idxs[demo_idx]]
            text_inputs.append(f"{x_idx} {theta_idx}: ")
        else:
            text_inputs.append(f"{x_idx}: ")
            
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
        'x_idx': x_idxs[shot],
    }
    

def load_dataset(
    shot,
    prompt_type,
    task_id, 
    seed = 123,
    data_mode = 'inference', # 'inference' or 'ft_train' or 'ft_test'
):
    print("========"*3)
    print(f'Loading the dataset for task {task_id}...')
    print(f'| task type: {task_dataframe[task_id]["task_type"]}')
    print(f'| x_space: {task_dataframe[task_id]["x_space"]}')
    print(f'| theta_space: {task_dataframe[task_id]["theta_space"]}')
    print(f'| prompt_type: {prompt_type}')
    print(f'| shot: {shot}')
    
    set_seed(seed)
    
    prompts_list = read_json(f"{root_dir}/load_datasets/prompts_list_{data_mode}.json")
    
    if data_mode in ['inference', 'ft_test']:
        data_loader = []
        for i in range(num_prompt_dict[data_mode]):
            item_inputs = prompts_list[i]
            if data_mode == 'inference':
                theta_input = item_inputs["theta_list"]
            else:
                theta_input = [item_inputs["theta_list"][i%len(item_inputs["theta_list"])] for i in range(shot+2)]
            
            input_dict = load_inputs(
                shot,
                prompt_type,
                task_id,
                item_inputs["x_list"],
                theta_input,
                task_dataframe[task_id]["x_list"],
                task_dataframe[task_id]["theta_list"],
            )
            input_dict['save_path'] = f"{i}_{input_dict['theta']}_{'_'.join(input_dict['x_list'])}"
            data_loader.append(input_dict)
    elif data_mode == 'ft_train':
        x_lists = list(permutations(prompts_list, shot + 1))
        theta_lists = list(permutations(prompts_list, 1))
        
        data_loader = []
        for x_list in x_lists:
            for theta_list in theta_lists:
                input_dict = load_inputs(
                    shot,
                    prompt_type,
                    task_id,
                    x_list,
                    [None for _ in range(shot+1)] + [theta_list[0]],
                    task_dataframe[task_id]["x_list"],
                    task_dataframe[task_id]["theta_list"],
                )
                data_loader.append(input_dict)
    else:
        raise NotImplementedError(f"Unknown data_mode: {data_mode}!")
            
    print('Done!')
    print("========"*3)
    return data_loader
        
def get_instruction(
    prompt_type, 
    gen_mode,
    task_id, 
    model,
):
    if prompt_type == 'instruct':
        return (instruction_dict[prompt_type][gen_mode][task_id], '')
    elif prompt_type == 'caption':
        return (instruction_dict[prompt_type][gen_mode], '')
    elif prompt_type == 'cot':
        return instruction_dict[prompt_type][gen_mode]
    elif prompt_type in ['default', 'misleading']:
        if model in instruction_dict['default'][gen_mode]:
            return instruction_dict['default'][gen_mode][model]
        else:
            raise NotImplementedError(f'{model} is not supported for {gen_mode} generation!')
    else:
        raise NotImplementedError(f'{prompt_type} is not supported!')
        
        
def get_prompt(
    text_inputs,
    image_inputs,
    prompt_type,
    task_id, 
    model,
    gen_mode, 
    history = None,
):
    if prompt_type in ['instruct', 'default', 'misleading']: # [-1,0,1]:
        query = {
            'text_inputs': text_inputs, 
            'image_inputs': image_inputs,
            'instruction': get_instruction(
                prompt_type, 
                gen_mode,
                task_id,
                model,
            )
        }
    elif prompt_type == 'caption': # -2:
        for i, image_path in enumerate(image_inputs):
            caption = find_caption(image_path)
            text_inputs.insert(
                2*i+1, 
                caption + ' '
            )
            
        query = {
            'text_inputs': text_inputs,
            'image_inputs': [],
            'instruction': get_instruction(
                prompt_type, 
                gen_mode,
                task_id,
                model,
            ),
            'call_mode': 'text',
        }
    elif prompt_type == 'cot':
        if gen_mode == 'general':
            query = {
                'text_inputs': text_inputs, 
                'image_inputs': image_inputs,
                'instruction': get_instruction(
                    prompt_type, 
                    gen_mode,
                    task_id,
                    model,
                ),
                'save_history': True,
            }
        else:
            instruction = get_instruction(
                prompt_type, 
                gen_mode,
                task_id,
                model,
            )
            
            query = {
                'text_inputs': text_inputs, 
                'image_inputs': image_inputs,
                'instruction': instruction,
                'history': history,
            }
            
    else:
        raise NotImplementedError(f"Unknown prompt_type: {prompt_type}!")
    return query 