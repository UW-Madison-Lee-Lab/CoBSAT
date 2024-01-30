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
):
    set_seed(123)
    
    if prompt_type in ['misleading', 'cot', 'caption']: 
        raise ValueError(f"Finetuning mode does not support prompt type {prompt_type}!")
    
    path_dict = get_ft_path(
        model,
        gen_mode,
        shot,
        prompt_type,
    )
    
    output_dir, data_path = path_dict['model'], path_dict['data']
    
    data_loader = {}
    for task_id in task_dataframe:
        data_loader[task_id] = load_dataset(
            shot,
            prompt_type = 'default',
            task_id = task_id,
            data_mode = 'ft_train',
        )
        
    if model == 'qwen':
        if gen_mode != 'text': raise ValueError(f"Incompatible gen_mode with {model}: {gen_mode}!")
        from load_models.call_qwen import ft_qwen, load_ft_qwen_prompt
        
        data_ft = []
        for task_id in task_dataframe:
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
    )
    