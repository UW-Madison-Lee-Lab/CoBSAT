import os, sys
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

from load_dataset import load_dataset
from configs import task_dataframe
from helper import save_json, get_ft_path, set_seed
import argparse

def ft_model(
    model, 
    shot,
    prompt_type,
    gen_mode,
    ft_mode = 'all', # 'all' or 'leave_one_out' 
    eval_task_theme = '', # color, background, style, action, texture 
):
    set_seed(123)
    
    if prompt_type in ['misleading', 'cot', 'caption']: 
        raise ValueError(f"Finetuning mode does not support prompt type {prompt_type}!")
    
    if (ft_mode == 'leave_one_out' and (not eval_task_theme)) or (ft_mode == 'all' and eval_task_theme):
        raise ValueError(f"ft_mode and eval_task_theme are incompatible!")
    
    path_dict = get_ft_path(
        model,
        gen_mode,
        shot,
        prompt_type,
    )
    
    output_dir, data_path = path_dict['model'], path_dict['data']
    
    data_loader = {}
    
    if model == 'qwen':
        include_output = False  
    elif model == 'seed':
        include_output = True
    else:
        raise ValueError(f"Unknown model: {model}")
    
    for task_id in task_dataframe:
        if ft_mode == 'leave_one_out':
            if task_dataframe[task_id]['task_name'].split('-')[0].lower() == eval_task_theme:
                continue 
        
        data_loader[task_id] = load_dataset(
            shot,
            prompt_type = 'default',
            task_id = task_id,
            data_mode = 'ft_train',
            include_output = include_output,
            ft_mode = ft_mode,
        )
        
    if model == 'qwen':
        if gen_mode != 'text': raise ValueError(f"Incompatible gen_mode with {model}: {gen_mode}!")
        from load_models.call_qwen import ft_qwen, load_ft_qwen_prompt
        
        data_ft = []
        for task_id in data_loader:
            for idx, item in enumerate(data_loader[task_id]):
                data_ft.append(load_ft_qwen_prompt(
                    task_id,
                    idx,
                    prompt_type,
                    item['text_inputs'],
                    item['image_inputs'],
                    item['x_idx'],
                    item['theta']
                ))
                
        save_json(data_ft, data_path)
        
        ft_qwen(
            data_path,
            output_dir,
        )
    elif model == 'seed':
        if gen_mode != 'image': raise ValueError(f"Incompatible gen_mode with {model}: {gen_mode}!")
        from load_models.call_seed import ft_seed
        
        data_ft = []
        for task_id in data_loader:
            data_ft.extend(data_loader[task_id])

        ft_seed(
            data_ft,
            output_dir,          
        )
    else:
        raise ValueError(f"Unknown model: {model}")

if '__main__' == __name__:
    parser = argparse.ArgumentParser("""
        Finetune the model using our dataset.
        
        Example Usage:
        $ CUDA_VISIBLE_DEVICES=7 CUDA_DEVICE_MAX_CONNECTIONS=1 python finetune_icl.py --model qwen --shot 2
    """)
    parser.add_argument('--model', type=str, default='qwen', help='model name')
    parser.add_argument('--shot', type=int, default=2, help='number of demonstrations')
    parser.add_argument('--prompt_type', type=str, default='default', help='prompt type')
    parser.add_argument('--gen_mode', type=str, default='text', help='generation mode')
    parser.add_argument('--ft_mode', type=str, default='all', choices = ['all', 'leave_one_out'], help='finetune mode')
    parser.add_argument('--eval_task_theme', type = str, default = '', choices = ['', 'color', 'background', 'style', 'action', 'texture'], help = 'task theme for evaluation')
    args = parser.parse_args()
    
    # print experiment configuration
    args_dict = vars(args)
    print("########"*3)
    print('## Experiment Setting:')
    print("########"*3)
    for key, value in args_dict.items():
        print(f"| {key}: {value}")
    
    ft_model(
        args.model,
        args.shot,
        args.prompt_type,
        args.gen_mode,
        args.ft_mode,
        args.eval_task_theme,
    )
    