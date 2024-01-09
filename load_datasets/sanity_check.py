# Using the command to activate the environment for llava
#### conda activate llava

import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from environment import TRANSFORMER_CACHE
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMER_CACHE

from load_models.call_llava import load_llava
from models.llava.llava.eval.run_llava import eval_model
import argparse, pandas as pd
from helper import set_seed
from configs import item_dict, task_types
from evaluation_icl import check_single_description, check_single_image
    
def check_text(task_type, llava_configs):
    
    category_space = {}
    category_space['detail'], category_space['obj'] = task_type.split('_')
    item_list = {
        'obj': item_dict[category_space['obj']],
        'detail': item_dict[category_space['detail']],
    }
    result_df = []
    for obj in item_list['obj']:
        folder_path = f"{root_dir}/datasets/{category_space['detail']}_{obj}"
        images = os.listdir(folder_path)
        for image in images:
            if not image.endswith('jpg'): continue
            
            ground_truth = {}
            ground_truth['detail'], ground_truth['obj'] = image.split('.')[0].split('_')
            
            try:
                image_path = f"{folder_path}/{image}"
                
                # llava for generating captions
                caption_prompt = f"Generate a clear description of the image <image-placeholder>. The description should include the object and details such as background, style, texture, color, action, etc, if applicable."
                caption = eval_model(
                    caption_prompt,
                    [image_path],
                    llava_configs['tokenizer'],
                    llava_configs['llava_model'],
                    llava_configs['image_processor'],
                    llava_configs['context_len'],
                    llava_configs['llava_args'],
                    device=llava_configs['device'],
                )
                
                row = check_single_description(
                    task_type,
                    image,
                    caption,
                    ground_truth,
                    llava_configs,
                )
            
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                print(f"Exception occurred: {type(e).__name__}, {e.args}")
                print(image)
                
                row = {
                    'image': image,
                    'caption': None,
                    'prompt_detail': None,
                    'prompt_obj': None,
                    'ground_truth_detail': None,
                    'ground_truth_obj': None,
                    'response_detail': None,
                    'response_obj': None,
                    'answer_detail': None,
                    'answer_obj': None,
                    'check_detail': None,
                    'check_obj': None,
                    'correct': None,
                }
                
            print(row)
            result_df.append(row)
            
            
    result_df = pd.DataFrame(result_df)
    df_path = f"{root_dir}/results/checks/text/{task_type}.csv"
    os.makedirs(os.path.dirname(df_path), exist_ok=True)
    result_df.to_csv(df_path)
        
                
def check_image(task_type, llava_configs):
    
    category_space = {}
    category_space['detail'], category_space['obj'] = task_type.split('_')
    item_list = {
        'obj': item_dict[category_space['obj']],
        'detail': item_dict[category_space['detail']],
    }
    
    result_df = []
    for obj in item_list['obj']:
        folder_path = f"{root_dir}/datasets/{category_space['detail']}_{obj}"
        images = os.listdir(folder_path)
        for image in images:
            if not image.endswith('jpg'): continue
            
            ground_truth = {}
            ground_truth['detail'], ground_truth['obj'] = image.split('.')[0].split('_')
            
            try:
                image_path = f"{folder_path}/{image}"
                
                row = check_single_image(
                    task_type,
                    image_path,
                    ground_truth,
                    llava_configs,
                )
                     
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                print(f"Exception occurred: {type(e).__name__}, {e.args}")
                print(image)
                row = {
                    'file_path': image,
                    'prompt_detail': None,
                    'prompt_obj': None,
                    'ground_truth_detail': None,
                    'ground_truth_obj': None,
                    'response_detail': None,
                    'response_obj': None,
                    'answer_detail': None,
                    'answer_obj': None,
                    'check_detail': None,
                    'check_obj': None,
                    'correct': None,
                }
            
            print(row)
            result_df.append(row)
            
    result_df = pd.DataFrame(result_df)
    df_path = f"{root_dir}/results/checks/image/{task_type}.csv"
    os.makedirs(os.path.dirname(df_path), exist_ok=True)
    result_df.to_csv(df_path)

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Generate image descriptions for the dataset')
    parser.add_argument('--mode', type=str, default = 'image', help='what check to do', choices = ['image', 'text'])
    parser.add_argument('--task_type', type=str, nargs='+', default = task_types, help='what task to check', choices = task_types)
    parser.add_argument('--device', type=str, default = 'cuda:0', help='what device to use')
    
    args = parser.parse_args()
    
    set_seed(123)
    llava_tokenizer, llava_model, llava_image_processor, llava_context_len, llava_args = load_llava(device=args.device)

    llava_configs = {
        'tokenizer': llava_tokenizer,
        'llava_model': llava_model,
        'image_processor': llava_image_processor,
        'context_len': llava_context_len,
        'llava_args': llava_args,
        'device': args.device,
    }
    
    if args.mode == 'text':
        for task_type in args.task_type:
            check_text(task_type, llava_configs)
    elif args.mode == 'image':
        for task_type in args.task_type:
            check_image(task_type, llava_configs)
        
    