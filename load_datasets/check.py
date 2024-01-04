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

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

llava_tokenizer, llava_model, llava_image_processor, llava_context_len, llava_args = load_llava(device='cuda:1')
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map='cuda:0', trust_remote_code=True).eval()
qwen_model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

def generate_single_caption(image_path):
    prompt = f"Generate a clear description of the image <image-placeholder>"
    caption = eval_model(
        prompt,
        [image_path],
        llava_tokenizer,
        llava_model,
        llava_image_processor,
        llava_context_len,
        llava_args,
        device='cuda',
    )
    
    return caption

def generate_captions(
    dataset_folder_path,
    overwrite = False,
    test = False,
):
    # recursively find all images
    for root, dirs, files in os.walk(dataset_folder_path):
        print(f"Processing {os.path.split(root)[0]}...")
        for file in files:
            if file.endswith(".jpg"):
                
                image_path = os.path.join(root, file)
                caption_path = image_path.replace('datasets', 'results/captions').replace('.jpg', '.txt')
                folder = os.path.dirname(caption_path)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                    
                if os.path.exists(caption_path) and not overwrite:
                    continue
            
                caption = generate_single_caption(image_path)
                print(f"Caption for {file}: {caption}")
                with open(caption_path, 'w') as f:
                    f.write(caption)
                    
                if test: break
    
def check_caption(task_type):
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
                caption_prompt = f"Generate a clear description of the image <image-placeholder>."
                caption = eval_model(
                    caption_prompt,
                    [image_path],
                    llava_tokenizer,
                    llava_model,
                    llava_image_processor,
                    llava_context_len,
                    llava_args,
                    device='cuda:1',
                )
                
                ground_truth = {}
                ground_truth['detail'], ground_truth['obj'] = image.split('.')[0].split('_')
                
                # two prompts
                prompts = {
                    'detail': f"What is the {category_space['detail']} (of the main object) in the image based on the description? Answer from the following options: ",
                    'obj': f"What is the main object in this image based on the description? Answer from the following options: ",
                } 
                
                checks, options, response = {}, {}, {}
                for mode in prompts:
                    for i, item in enumerate(item_list[mode]):
                        if item in image:
                            prompts[mode] += f"({i+1}){item2word.get(item, item)}"
                    prompts[mode] += ". Answer the number only and do not include any other texts (e.g., 1)."
                    
                    # qwen for answering questions
                    qwen_query = qwen_tokenizer.from_list_format([
                        {'text': f"Given an image with description '{caption}'. {prompts[mode]}"}
                    ])
                    response[mode] = qwen_model.chat(
                        qwen_tokenizer,
                        query = qwen_query,
                        history = None,
                    )[0]
                    
                    try:
                        options[mode] = int(''.join(filter(str.isdigit, response[mode])))
                    except KeyboardInterrupt:
                        exit()
                    except Exception as e:
                        print(e)
                        
                        options[mode] = -1
                        checks[mode] = False
                        continue
                    
                    true_label = item_list[mode].index(ground_truth[mode])+1
                    
                    if options[mode] == true_label: 
                        checks[mode] = True
                    else:
                        checks[mode] = False
                        
                row = {
                    'image': image,
                    'caption': caption,
                    'ground_truth_detail': ground_truth['detail'],
                    'ground_truth_obj': ground_truth['obj'],
                    'response_detail': response['detail'],
                    'response_obj': response['obj'],
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
                continue
            
    result_df = pd.DataFrame(result_df)
    df_path = f"{root_dir}/results/checks/caption/{task_type}.csv"
    os.makedirs(os.path.dirname(df_path), exist_ok=True)
    result_df.to_csv(df_path)
        
                
def check_image(task_type):
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
                
                checks, options, response = {}, {}, {}
                for mode in prompts:
                    for i, item in enumerate(item_list[mode]):
                        if item in image:
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
                        device='cuda',
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
                    
                    true_label = item_list[mode].index(ground_truth[mode])+1
                    
                    if options[mode] == true_label: 
                        checks[mode] = True
                    else:
                        checks[mode] = False
                        
                row = {
                    'image': image,
                    'ground_truth_detail': ground_truth['detail'],
                    'ground_truth_obj': ground_truth['obj'],
                    'response_detail': response['detail'],
                    'response_obj': response['obj'],
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
                continue
            
    result_df = pd.DataFrame(result_df)
    df_path = f"{root_dir}/results/checks/image/{task_type}.csv"
    os.makedirs(os.path.dirname(df_path), exist_ok=True)
    result_df.to_csv(df_path)

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Generate image descriptions for the dataset')
    parser.add_argument('--mode', type=str, default = 'image', help='what check to do', choices = ['image', 'caption'])
    parser.add_argument('--overwrite', type=int, default = 0, help='overwrite existing captions or not', choices = [0,1])
    parser.add_argument('--task_type', type=str, nargs='+', default = task_types, help='what task to check', choices = task_types)
    
    args = parser.parse_args()
    
    set_seed(123)
    
    if args.mode == 'caption':
        # generate_captions(
        #     f"{root_dir}/datasets",
        #     overwrite = args.overwrite,
        #     test = args.test,
        # )
        for task_type in args.task_type:
            check_caption(task_type)
    elif args.mode == 'image':
        for task_type in args.task_type:
            check_image(task_type)
        
    