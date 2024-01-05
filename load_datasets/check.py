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
from configs import item_dict, task_types, item2word
    
def check_caption(task_type, llava_configs):
    llava_tokenizer = llava_configs['tokenizer']
    llava_model = llava_configs['model']
    llava_image_processor = llava_configs['image_processor']
    llava_context_len = llava_configs['context_len']
    llava_args = llava_configs['args']
    llava_device = llava_configs['device']
    
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
            try:
                if image.startswith('.'): continue # skip .DS_Store
                image_path = f"{folder_path}/{image}"
                
                # llava for generating captions
                caption_prompt = f"Generate a clear description of the image <image-placeholder>. The description should include the object and details such as background, weather, texture, color, action, etc, if applicable."
                caption = eval_model(
                    caption_prompt,
                    [image_path],
                    llava_tokenizer,
                    llava_model,
                    llava_image_processor,
                    llava_context_len,
                    llava_args,
                    device=llava_device,
                )
                
                ground_truth = {}
                ground_truth['detail'], ground_truth['obj'] = image.split('.')[0].split('_')
                
                # two prompts
                prompts = {
                    'detail': f"What is the {category_space['detail']} (of the main object) in the image based on the description? Answer from the following options: ",
                    'obj': f"What is the main object in this image based on the description? Answer from the following options: ",
                } 
                
                checks, options, response, true_labels = {}, {}, {}, {}
                for mode in prompts:
                    for i, item in enumerate(item_list[mode]):
                        if item in image:
                            prompts[mode] += f"({i+1}){item2word.get(item, item)}"
                    prompts[mode] += ". Answer the number only and do not include any other texts (e.g., 1)."
                    
                    response[mode] = eval_model(
                        prompts[mode],
                        [image_path],
                        llava_tokenizer,
                        llava_model,
                        llava_image_processor,
                        llava_context_len,
                        llava_args,
                        device=llava_device,
                    )
                    
                    try:
                        options[mode] = int(''.join(filter(str.isdigit, response[mode])))
                    except KeyboardInterrupt:
                        exit()
                    except Exception as e:
                        print(e)
                        
                        options[mode] = -1
                        checks[mode] = False
                        continue
                    
                    true_labels[mode] = item_list[mode].index(ground_truth[mode])+1
                    
                    if options[mode] == true_labels[mode]: 
                        checks[mode] = True
                    else:
                        checks[mode] = False
                        
                row = {
                    'image': image,
                    'caption': caption,
                    'prompt_detail': prompts['detail'],
                    'prompt_obj': prompts['obj'],
                    'ground_truth_detail': ground_truth['detail'],
                    'ground_truth_obj': ground_truth['obj'],
                    'response_detail': response['detail'],
                    'response_obj': response['obj'],
                    'true_label_detail': true_labels['detail'],
                    'true_label_obj': true_labels['obj'],
                    'answer_detail': options['detail'],
                    'answer_obj': options['obj'],
                    'check_detail': checks['detail'],
                    'check_obj': checks['obj'],
                    'correct': checks['detail'] and checks['obj'],
                }
                result_df.append(row)
                print(row)
            
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                print(e)
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
                result_df.append(row)
            
            
    result_df = pd.DataFrame(result_df)
    df_path = f"{root_dir}/results/checks/caption/{task_type}.csv"
    os.makedirs(os.path.dirname(df_path), exist_ok=True)
    result_df.to_csv(df_path)
        
                
def check_image(task_type, llava_configs):
    llava_tokenizer = llava_configs['tokenizer']
    llava_model = llava_configs['model']
    llava_image_processor = llava_configs['image_processor']
    llava_context_len = llava_configs['context_len']
    llava_args = llava_configs['args']
    llava_device = llava_configs['device']
    
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
            try:
                if image.startswith('.'): continue # skip .DS_Store
                image_path = f"{folder_path}/{image}"
                
                ground_truth = {}
                ground_truth['detail'], ground_truth['obj'] = image.split('.')[0].split('_')
                
                # two prompts
                prompts = {
                    'detail': f"What is the {category_space['detail']} (of the main object) in this image? Answer from the following options: ",
                    'obj': f"What is the main object in this image? Answer from the following options: ",
                } 
                
                checks, options, response, true_labels = {}, {}, {}, {}
                for mode in prompts:
                    for i, item in enumerate(item_list[mode]):
                        prompts[mode] += f"({i+1}){item2word.get(item, item)}"
                    prompts[mode] += ". Answer the number only and do not include any other texts (e.g., 1)."
                    
                    # llava
                    response[mode] = eval_model(
                        prompts[mode],
                        [image_path],
                        llava_tokenizer,
                        llava_model,
                        llava_image_processor,
                        llava_context_len,
                        llava_args,
                        device=llava_device,
                    )
                    
                    try:
                        options[mode] = int(''.join(filter(str.isdigit, response[mode])))
                    except KeyboardInterrupt:
                        exit()
                    except Exception as e:
                        print(e)
                        
                        options[mode] = -1
                        checks[mode] = False
                        continue
                    
                    true_labels[mode] = item_list[mode].index(ground_truth[mode])+1
                    
                    if options[mode] == true_labels[mode]: 
                        checks[mode] = True
                    else:
                        checks[mode] = False
                        
                row = {
                    'image': image,
                    'prompt_detail': prompts['detail'],
                    'prompt_obj': prompts['obj'],
                    'ground_truth_detail': ground_truth['detail'],
                    'ground_truth_obj': ground_truth['obj'],
                    'response_detail': response['detail'],
                    'response_obj': response['obj'],
                    'true_label_detail': true_labels['detail'],
                    'true_label_obj': true_labels['obj'],
                    'answer_detail': options['detail'],
                    'answer_obj': options['obj'],
                    'check_detail': checks['detail'],
                    'check_obj': checks['obj'],
                    'correct': checks['detail'] and checks['obj'],
                }
                result_df.append(row)
                print(row)
            
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                print(e)
                print(image)
                row = {
                    'image': image,
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
                result_df.append(row)
            
    result_df = pd.DataFrame(result_df)
    df_path = f"{root_dir}/results/checks/image/{task_type}.csv"
    os.makedirs(os.path.dirname(df_path), exist_ok=True)
    result_df.to_csv(df_path)

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Generate image descriptions for the dataset')
    parser.add_argument('--mode', type=str, default = 'image', help='what check to do', choices = ['image', 'caption'])
    parser.add_argument('--task_type', type=str, nargs='+', default = task_types, help='what task to check', choices = task_types)
    parser.add_argument('--device', type=str, default = 'cuda:0', help='what device to use')
    
    args = parser.parse_args()
    
    set_seed(123)
    llava_tokenizer, llava_model, llava_image_processor, llava_context_len, llava_args = load_llava(device=args.device)
    llava_configs = {
        'tokenizer': llava_tokenizer,
        'model': llava_model,
        'image_processor': llava_image_processor,
        'context_len': llava_context_len,
        'args': llava_args,
        'device': args.device,
    }
    
    if args.mode == 'caption':
        for task_type in args.task_type:
            check_caption(task_type, llava_configs)
    elif args.mode == 'image':
        for task_type in args.task_type:
            check_image(task_type, llava_configs)
        
    