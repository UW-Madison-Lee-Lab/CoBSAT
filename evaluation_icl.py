import os, sys
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

from load_models.call_llava import load_llava, eval_model as infer_llava
import argparse, pandas as pd, wandb, torch, numpy as np
from helper import set_seed, read_json
from configs import task_dataframe, item_dict, item2word, supported_models
from load_dataset import load_dataset
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def get_clip_similarity(clip_model, batch_true_embeds, pred_embeds):
    logit_scale = clip_model.logit_scale.exp()
    # note that the text_embeds and img_embeds are already normalized
    # torch.matmul here is computing the cosine similarity
    clip_similarity = torch.matmul(batch_true_embeds, pred_embeds.t()) * logit_scale
    clip_similarity = clip_similarity.t()[0].detach().cpu().numpy()
    for i in range(len(clip_similarity)):
        clip_similarity[i] = max(float(clip_similarity[i]), 0)
    return clip_similarity

def eval_clip(
    file_path, 
    ground_truth,
    clip_model, 
    clip_processor,
    item_list,
    true_labels,
    existing_csv,
    eval_mode,
):
    if not (existing_csv is None):
        row = existing_csv[existing_csv['file_path'] == file_path]
        
        success_flag = True
        output_dict = {}
        keys = [
            'clip_similarity_detail', 
            'clip_similarity_obj',
            'clip_similarity_overall',
            'clip_check_detail',
            'clip_check_obj',
            'clip_correct',
        ]
        for key in keys:
            if key in existing_csv.columns:
                output_dict[key] = row[key].item()
                if key in [
                    'clip_similarity_detail', 
                    'clip_similarity_obj',
                    'clip_similarity_overall',
                ]:
                    output_dict[key] = float(output_dict[key])
                else:
                    output_dict[key] = output_dict[key] == 'True'
            else:
                success_flag = False
                break
        if success_flag: return output_dict
        
    detail, obj = ground_truth['detail'], ground_truth['obj']
    detail_list, obj_list = item_list['detail'], item_list['obj']
    
    if eval_mode == 'image':
        image = Image.open(file_path).convert("RGB")
        inputs = clip_processor(
            text=[detail, obj, f"{detail} {obj}"] + obj_list + detail_list, 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(clip_model.device)
        outputs = clip_model(**inputs)
        clip_similarity = get_clip_similarity(
            clip_model,
            outputs.text_embeds,
            outputs.image_embeds,
        )
    elif eval_mode == 'text':
        description = read_json(file_path)['description']
        batch_inputs = clip_processor(
            text=[description[:200], detail, obj, f"{detail} {obj}"] + obj_list + detail_list, 
            return_tensors="pt", 
            padding=True,
        ).to(clip_model.device)
        batch_outputs = clip_model.get_text_features(**batch_inputs)
        batch_outputs = batch_outputs / batch_outputs.norm(dim=1, keepdim=True)
        description_embeds = batch_outputs[[0]]
        batch_true_embeds = batch_outputs[1:]
        clip_similarity = get_clip_similarity(
             clip_model,
             batch_true_embeds,
             description_embeds,
        )

    # similarity between the generated image and the ground truth
    output_dict = {
        'clip_similarity_detail': clip_similarity[0],
        'clip_similarity_obj': clip_similarity[1],
        'clip_similarity_overall': clip_similarity[2],
    }
    # similarity between the generated image the all possible objects and details
    pred_obj = np.argmax(clip_similarity[3:(3+len(obj_list))]) + 1
    pred_detail = np.argmax(clip_similarity[(3+len(obj_list)):(3+len(obj_list)+len(detail_list))]) + 1
    
    output_dict['clip_check_obj'] = pred_obj == true_labels['obj']
    output_dict['clip_check_detail'] = pred_detail == true_labels['detail']
    output_dict['clip_correct'] = output_dict['clip_check_obj'] and output_dict['clip_check_detail']
    
    return output_dict

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
    

def eval_llava(
    category_space,
    item_list,
    file_path,
    true_labels,
    llava_configs,
    existing_csv,
    eval_mode,
):
    if not (existing_csv is None):
        row = existing_csv[existing_csv['file_path'] == file_path]
        
        success_flag = True
        output_dict = {}
        keys = [
            'prompt_detail',
            'prompt_obj',
            'response_detail',
            'response_obj',
            'answer_detail',
            'answer_obj',
            'check_detail',
            'check_obj',
            'correct',
        ]
        
        for key in keys:
            if key in existing_csv.columns:
                output_dict[key] = row[key].item()
                if key in ['check_detail', 'check_obj', 'correct']: 
                    output_dict[key] = output_dict[key] == 'True'
            else:
                success_flag = False
                break
        if success_flag: return output_dict
        
    # two prompts
    caption = None
    if eval_mode == 'text':
        caption = read_json(file_path)['description']
        
    prompts = get_eval_prompt(
        category_space,
        item_list,
        eval_mode,
        caption,
    )
        
    checks, options, response = {}, {}, {}
    for mode in prompts:
        if eval_mode == 'image':
            response[mode] = infer_llava(
                prompts[mode],
                [file_path],
                llava_configs['tokenizer'],
                llava_configs['llava_model'],
                llava_configs['image_processor'],
                llava_configs['context_len'],
                llava_configs['llava_args'],
                device=llava_configs['device'],
            )
        elif eval_mode == 'text':
            response[mode] = infer_llava(
                prompts[mode],
                [],
                llava_configs['tokenizer'],
                llava_configs['llava_model'],
                llava_configs['image_processor'],
                llava_configs['context_len'],
                llava_configs['llava_args'],
                device=llava_configs['device'],
            )
        else:
            raise NotImplementedError(f"Unknown eval_mode: {eval_mode}!")
        
        response_number = ''.join(filter(str.isdigit, response[mode]))
        if response_number:
            options[mode] = int(response_number)
        else:
            options[mode] = -1
        
        if options[mode] == true_labels[mode]: 
            checks[mode] = True
        else:
            checks[mode] = False
            
    output_dict = {
        'prompt_detail': prompts['detail'],
        'prompt_obj': prompts['obj'],
        'response_detail': response['detail'],
        'response_obj': response['obj'],
        'answer_detail': options['detail'],
        'answer_obj': options['obj'],
        'check_detail': checks['detail'],
        'check_obj': checks['obj'],
        'correct': checks['detail'] and checks['obj'],
    }

    return output_dict

def check_single_output(
    task_type,
    file_path,
    ground_truth,
    llava_configs,
    existing_csv,
    eval_mode,
):
    category_space = {}
    category_space['detail'], category_space['obj'] = task_type.split('_')
    item_list = {
        'obj': item_dict[category_space['obj']],
        'detail': item_dict[category_space['detail']],
    }
    
    true_labels = {}
    if not os.path.exists(file_path):
        row = {
            'file_path': file_path,
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
            'check_detail': 0,
            'check_obj': 0,
            'correct': False,
            'clip_similarity_detail': 0,
            'clip_similarity_obj': 0,
            'clip_similarity_overall': 0,
            'clip_check_detail': False,
            'clip_check_obj': False,
            'clip_correct': False,
        }
        if eval_mode == 'text': row['caption'] = None
    else:
        for mode in ['detail', 'obj']:
            true_labels[mode] = item_list[mode].index(ground_truth[mode])+1
        
        # use llava to evaluate the quality of the generated images

        llava_output = eval_llava(
            category_space,
            item_list,
            file_path,
            true_labels,
            llava_configs,
            existing_csv, 
            eval_mode,
        )
         
        # use clip to evaluate the quality of the generated images
        clip_output = eval_clip(
            file_path,
            ground_truth,
            clip_model,
            clip_processor,
            item_list,
            true_labels,
            existing_csv,
            eval_mode,
        )
                
        row = {
            'file_path': file_path,
            'prompt_detail': llava_output['prompt_detail'],
            'prompt_obj': llava_output['prompt_obj'],
            'ground_truth_detail': ground_truth['detail'],
            'ground_truth_obj': ground_truth['obj'],
            'response_detail': llava_output['response_detail'],
            'response_obj': llava_output['response_obj'],
            'true_label_detail': true_labels['detail'],
            'true_label_obj': true_labels['obj'],
            'answer_detail': llava_output['answer_detail'],
            'answer_obj': llava_output['answer_obj'],
            'check_detail': llava_output['check_detail'],
            'check_obj': llava_output['check_obj'],
            'correct': llava_output['correct'],
            'clip_similarity_detail': clip_output['clip_similarity_detail'],
            'clip_similarity_obj': clip_output['clip_similarity_obj'],
            'clip_similarity_overall': clip_output['clip_similarity_overall'],
            'clip_check_detail': clip_output['clip_check_detail'],
            'clip_check_obj': clip_output['clip_check_obj'],
            'clip_correct': clip_output['clip_correct'],
        }
        
        if eval_mode == 'text': row['caption'] = read_json(file_path)['description']
        
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
    
    if misleading == -1:
        misleading_flag = "_i"
    elif misleading == 0:
        misleading_flag = ''
    elif misleading == 1:
        misleading_flag = "_m"
    else:
        raise NotImplementedError(f"Unknown misleading: {misleading}!")
    
    csv_file_path = f"{root_dir}/results/evals/{model}_{eval_mode}/shot_{shot}{misleading_flag}/task_{task_id}_summary.csv"
    existing_csv = None
    if os.path.exists(csv_file_path) and (not overwrite): existing_csv = pd.read_csv(csv_file_path)
    
    if log_wandb: # log the data into wandb
        wandb_config = {
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
        }
        
        # first check whether there exists a run with the same configuration
        api = wandb.Api(timeout=300)
        runs = api.runs("lee-lab-uw-madison/micl")
        find_existing_run = None
        for run in runs:
            run_config_list = {k: v for k,v in run.config.items() if not k.startswith('_')}
            this_run = True
            for key in wandb_config:
                if (not key in run_config_list) or (run_config_list[key] != wandb_config[key]): 
                    this_run = False
                    break
            if this_run: 
                find_existing_run = run
                print("########"*3)
                print(f"Find existing run in wandb: {run.name}")
                print("########"*3)
                break
            
        # initialize wandb
        if find_existing_run is None:
            wandb.init(
                project = 'micl',
                entity = 'lee-lab-uw-madison',
                config = wandb_config,
            )
        
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
        'valid_count':0,
        'clip_correct': 0,
        'clip_similarity_overall': 0,
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
        elif eval_mode == 'image':
            file_path = f"{folder_path}/{input_dict['save_path']}.jpg"            
        else:
            raise NotImplementedError(f"Unknown eval_mode: {eval_mode}!")
        
        row = check_single_output(
            task_type,
            file_path,
            ground_truth,
            llava_configs,
            existing_csv,
            eval_mode,
        )
        
        row['check_textual'] = row[f"check_{type_dict['x']}"]
        row['check_visual'] = row[f"check_{type_dict['theta']}"]
        
        for key in ['detail', 'obj', 'textual', 'visual']:
            checks[key] += row[f"check_{key}"]
        checks['overall'] += row['correct']
        checks['valid_count'] += 1
        checks['clip_similarity_overall'] += row['clip_similarity_overall']
        checks['clip_correct'] += row['clip_correct']
        
        result_df.append(row)
        
    checks['clip_similarity_overall'] /= checks['valid_count']
    checks['clip_correct'] /= checks['valid_count']
        
    summary_row = {
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
        'clip_similarity_detail': None,
        'clip_similarity_obj': None,
        'clip_similarity_overall': checks['clip_similarity_overall'],
        'clip_check_detail': None,
        'clip_check_obj': None,
        'clip_correct': checks['clip_correct'],
    }
    if eval_mode == 'text': summary_row['caption'] = None
    result_df.append(summary_row) 
        
    result_df = pd.DataFrame(result_df)
    
    if os.path.dirname(csv_file_path) != '':
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    result_df.to_csv(csv_file_path, index = False)
    
    if log_wandb:
        if find_existing_run is None:
            wandb.log(checks)
            wandb.finish()
        else:
            print('Previous run statistics:')
            for k,v in find_existing_run.summary.items():
                if not k.startswith('_'):
                    print(f"| {k}: {v}")
            for key in checks:
                find_existing_run.summary[key] = checks[key]
            find_existing_run.summary.update() # update the summary
            print('Updated run statistics:')
            for k,v in find_existing_run.summary.items():
                if not k.startswith('_'):
                    print(f"| {k}: {v}")
    

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description = 'Evaluate the results using LLaVA.')
    parser.add_argument('--model', type = str, default = 'qwen', choices = supported_models, help = 'model')
    parser.add_argument('--task_id', type = int, nargs = '+', default = list(task_dataframe.keys()), help = 'task id')
    parser.add_argument('--shot', type = int, nargs = '+', default = [2,4,6,8], help = 'shot')
    parser.add_argument('--misleading', type = int, nargs = '+', default = [0,1], help = 'misleading', choices = [-1,0,1])
    parser.add_argument('--device', type = str, default = 'cuda', help = 'device')
    parser.add_argument('--seed', type = int, default = 123, help = 'seed')
    parser.add_argument('--wandb', type = int, default = 1, help = 'whether log the results using wandb', choices = [0,1])
    parser.add_argument('--overwrite', type = int, default = 0, help = 'whether overwrite the existing results', choices = [0,1])
    parser.add_argument('--eval_mode', type = str, default = 'text', help = 'evaluation mode', choices = ['text', 'image'])
    parser.add_argument('--max_file_count', type = int, default = 1000, help = 'max file count')
    
    args = parser.parse_args()
    
    # print experiment configuration
    args_dict = vars(args)
    print("########"*3)
    print('## Experiment Setting:')
    print("########"*3)
    for key, value in args_dict.items():
        print(f"| {key}: {value}")
    
    # load llava
    tokenizer, llava_model, image_processor, context_len, llava_args = load_llava(device = args.device)
    llava_configs = {
        'tokenizer': tokenizer,
        'llava_model': llava_model,
        'image_processor': image_processor,
        'context_len': context_len,
        'llava_args': llava_args,
        'device': args.device,
    }
    
    # load clip to device
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(args.device)
    
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
                    max_file_count = args.max_file_count,
                    log_wandb = args.wandb,
                    overwrite = args.overwrite,
                    eval_mode = args.eval_mode,
                )