import os, sys
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

from load_models.call_llava import load_llava, eval_model as eval_llava
import argparse, pandas as pd, wandb
from helper import set_seed, read_json
from configs import task_dataframe, item_dict, item2word, supported_models
from load_dataset import load_dataset
from tqdm import tqdm

def get_eval_prompt(
    category_space,
    item_list,
    mode,
    caption = None,
):
    
    if mode == 'image': 
        prompts = {
            'detail': f"What is the {category_space['detail']} (of the main object) in this image? Answer from the following options: ",
            'obj': f"What is the main object in this image? Answer from the following options: ",
        }
    elif mode == 'text':
        if caption is None:
            raise Exception("Variable caption should not be None for text mode!")
        
        prompts = {
            'detail': f"Image caption: {caption}. What is the {category_space['detail']} (of the main object) in the image based on the description? Answer from the following options: ",
            'obj': f"Image caption: {caption}. What is the main object in this image based on the description? Answer from the following options: ",
        }
    else:
        raise Exception(f"Unknown mode: {mode}!")
    
    for mode in prompts:
        for i, item in enumerate(item_list[mode]):
            prompts[mode] += f" ({i+1}){item2word.get(item, item)}"
        prompts[mode] += ". Answer the number only and do not include any other texts (e.g., 1)."
        
    return prompts

def check_single_description(
    task_type,
    file_path,
    caption,
    ground_truth,
    llava_configs,
):  
    category_space = {}
    category_space['detail'], category_space['obj'] = task_type.split('_')
    item_list = {
        'obj': item_dict[category_space['obj']],
        'detail': item_dict[category_space['detail']],
    }
    
    # two prompts
    prompts = get_eval_prompt(
        category_space,
        item_list,
        'text',
        caption,
    )
    
    checks, options, response, true_labels = {}, {}, {}, {}
    for mode in prompts:
        
        response[mode] = eval_llava(
            prompts[mode],
            [],
            llava_configs['tokenizer'],
            llava_configs['llava_model'],
            llava_configs['image_processor'],
            llava_configs['context_len'],
            llava_configs['llava_args'],
            device=llava_configs['device'],
        )
        
        response_number = ''.join(filter(str.isdigit, response[mode]))
        if response_number:
            options[mode] = int(response_number)
        else:
            options[mode] = -1
        
        true_labels[mode] = item_list[mode].index(ground_truth[mode])+1
        
        if options[mode] == true_labels[mode]: 
            checks[mode] = True
        else:
            checks[mode] = False
 
    row = {
        'file_path': file_path,
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
    
    return row

def check_single_image(
    task_type,
    file_path,
    ground_truth,
    llava_configs,
):
    category_space = {}
    category_space['detail'], category_space['obj'] = task_type.split('_')
    item_list = {
        'obj': item_dict[category_space['obj']],
        'detail': item_dict[category_space['detail']],
    }
    
    # two prompts
    prompts = get_eval_prompt(
        category_space,
        item_list,
        'image',
        None,
    )
    
    checks, options, response, true_labels = {}, {}, {}, {}
    for mode in prompts:
        
        response[mode] = eval_llava(
            prompts[mode],
            [file_path],
            llava_configs['tokenizer'],
            llava_configs['llava_model'],
            llava_configs['image_processor'],
            llava_configs['context_len'],
            llava_configs['llava_args'],
            device=llava_configs['device'],
        )
        
        response_number = ''.join(filter(str.isdigit, response[mode]))
        if response_number:
            options[mode] = int(response_number)
        else:
            options[mode] = -1
        
        true_labels[mode] = item_list[mode].index(ground_truth[mode])+1
        
        if options[mode] == true_labels[mode]: 
            checks[mode] = True
        else:
            checks[mode] = False
            
    row = {
        'file_path': file_path,
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
    
    return row

def eval(
    task_id,
    shot,
    misleading,
    model,
    llava_configs,
    seed,
    max_file_count = 1000,
    log_wandb = False,
    overwrite = False,
    eval_mode = 'text',
):
    task_type = task_dataframe[task_id]['task_type']
    category_space = {}
    category_space['detail'], category_space['obj'] = task_type.split('_')
    
    if task_dataframe[task_id]['x_space'] in ['animal', 'object']:
        type_dict = {
            'x': 'obj', 'theta': 'detail',
            'obj': 'x', 'detail': 'theta',
        }
    else:
        type_dict = {
            'x': 'detail', 'theta': 'obj',
            'obj': 'theta', 'detail': 'x',
        }
    
    if log_wandb:
        wandb.init(
            project = 'micl',
            entity = 'lee-lab-uw-madison',
            config = {
                'task_id': task_id,
                'shot': shot,
                'misleading': misleading,
                'model': model,
                'seed': seed,
                'stage': 'eval',
                'file_type': eval_mode,
                'task_type': task_type,
                'x_space': task_dataframe[task_id]['x_space'],
                'theta_space': task_dataframe[task_id]['theta_space'],
            },
        )
        
    misleading_flag = "_m" if misleading else ""
    csv_file_path = f"{root_dir}/results/evals/{model}_{eval_mode}/shot_{shot}{misleading_flag}/task_{task_id}_summary.csv"
    if os.path.exists(csv_file_path) and not overwrite:
        print('The evaluation results already exist.')
        return
        
    set_seed(seed)
    
    data_loader = load_dataset(
        shot,
        misleading,
        task_id,
        max_file_count,
    )
    
    base_path = f"{root_dir}/results/exps/{model}_{eval_mode}/shot_{shot}{misleading_flag}"
    folder_path = f"{base_path}/task_{task_id}"
    if not os.path.exists(folder_path):
        raise Exception(f"Folder {folder_path} does not exist.")
    
    result_df = []
    checks = {
        'detail':0,
        'obj':0,
        'textual':0,
        'visual':0,
        'overall':0,
        'valid_count':0
    }
    
    for count in tqdm(range(max_file_count), desc = f"Evaluating {model}_{eval_mode}/shot_{shot}{misleading_flag}/task_{task_id}"):
            
        input_dict = data_loader[count]
        input_dict['x'] = input_dict['x_list'][-1]
        
        ground_truth = {
            'detail': input_dict[type_dict['detail']],
            'obj': input_dict[type_dict['obj']],
        }
        
        if eval_mode == 'text':
            file_path = f"{folder_path}/{input_dict['save_path']}.json"
            output_dict = read_json(file_path)
        
            row = check_single_description(
                task_type,
                file_path,
                output_dict['description'],
                ground_truth,
                llava_configs,
            )
        elif eval_mode == 'image':
            file_path = f"{folder_path}/{input_dict['save_path']}.jpg"
            row = check_single_image(
                task_type,
                file_path,
                ground_truth,
                llava_configs,
            )
            
        else:
            raise NotImplementedError(f"Unknown eval_mode: {eval_mode}!")
        
        row['check_textual'] = row[f"check_{type_dict['x']}"]
        row['check_visual'] = row[f"check_{type_dict['theta']}"]
        
        for key in ['detail', 'obj', 'textual', 'visual']:
            checks[key] += row[f"check_{key}"]
        checks['overall'] += row['correct']
        checks['valid_count'] += 1
        
        result_df.append(row)
        
    if eval_mode == 'text':
        result_df.append({
            'file_path': 'SUMMARY',
            'caption': f"Valid Count: {checks['valid_count']}",
            'prompt_detail': None,
            'prompt_obj': None,
            'ground_truth_detail': None,
            'ground_truth_obj': None,
            'response_detail': None,
            'response_obj': None,
            'true_label_detail': None,
            'true_label_obj': None,
            'answer_detail': None,
            'answer_obj': None,
            'check_detail': checks['detail'],
            'check_obj': checks['obj'],
            'check_textual': checks['textual'],
            'check_visual': checks['visual'],
            'correct': checks['overall'],
        }) 
    elif eval_mode == 'image':
        result_df.append({
            'file_path': 'SUMMARY',
            'prompt_detail': f"Valid Count: {checks['valid_count']}",
            'prompt_obj': None,
            'ground_truth_detail': None,
            'ground_truth_obj': None,
            'response_detail': None,
            'response_obj': None,
            'true_label_detail': None,
            'true_label_obj': None,
            'answer_detail': None,
            'answer_obj': None,
            'check_detail': checks['detail'],
            'check_obj': checks['obj'],
            'check_textual': checks['textual'],
            'check_visual': checks['visual'],
            'correct': checks['overall'],
        }) 
        
    result_df = pd.DataFrame(result_df)
    
    if os.path.dirname(csv_file_path) != '':
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    result_df.to_csv(csv_file_path, index = False)
    
    if log_wandb:
        wandb.log(checks)
        wandb.finish()
    

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description = 'Evaluate the results using LLaVA.')
    parser.add_argument('--model', type = str, default = 'qwen', choices = supported_models, help = 'model')
    parser.add_argument('--task_id', type = int, nargs = '+', default = list(task_dataframe.keys()), help = 'task id')
    parser.add_argument('--shot', type = int, nargs = '+', default = [4,6,8], help = 'shot')
    parser.add_argument('--misleading', type = int, nargs = '+', default = [0,1], help = 'misleading', choices = [0,1])
    parser.add_argument('--device', type = str, default = 'cuda', help = 'device')
    parser.add_argument('--seed', type = int, default = 123, help = 'seed')
    parser.add_argument('--wandb', type = int, default = 1, help = 'whether log the results using wandb', choices = [0,1])
    parser.add_argument('--overwrite', type = int, default = 0, help = 'whether overwrite the existing results', choices = [0,1])
    parser.add_argument('--eval_mode', type = str, default = 'text', help = 'evaluation mode', choices = ['text', 'image'])
    
    args = parser.parse_args()
    
    # print experiment configuration
    args_dict = vars(args)
    print("########"*3)
    print('## Experiment Setting:')
    print("########"*3)
    for key, value in args_dict.items():
        print(f"| {key}: {value}")
    
    tokenizer, llava_model, image_processor, context_len, llava_args = load_llava(device = args.device)
    llava_configs = {
        'tokenizer': tokenizer,
        'llava_model': llava_model,
        'image_processor': image_processor,
        'context_len': context_len,
        'llava_args': llava_args,
        'device': args.device,
    }
    
    for task_id in args.task_id:
        for shot in args.shot:
            for misleading in args.misleading:
                eval(
                    task_id,
                    shot,
                    misleading,
                    args.model,
                    llava_configs,
                    args.seed,
                    max_file_count = 1000,
                    log_wandb = args.wandb,
                    overwrite = args.overwrite,
                    eval_mode = args.eval_mode,
                )