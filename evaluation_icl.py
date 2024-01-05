import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from load_models.call_llava import load_llava, eval_model as eval_llava
import argparse
from helper import set_seed, save_json, read_json
from configs import task_dataframe, item_dict, item2word
from load_dataset import load_dataset
from tqdm import tqdm

def eval_description(
    task_id,
    shot,
    misleading,
    model,
    llava_configs,
    seed,
    max_file_count = 1000,
):
    set_seed(seed)
    
    task_type = task_dataframe[task_id]['task_type']
    category_space = {}
    category_space['detail'], category_space['obj'] = task_type.split('_')
    item_list = {
        'obj': item_dict[category_space['obj']],
        'detail': item_dict[category_space['detail']],
    }
    
    misleading_flag = "_m" if misleading else ""
    base_path = f"{root_dir}/results/exps/{model}_text/shot_{shot}{misleading_flag}"
    folder_path = f"{base_path}/task_{task_id}"
    if not os.path.exists(folder_path):
        raise Exception(f"Folder {folder_path} does not exist.")
    
    data_loader = load_dataset(
        shot,
        misleading,
        task_id,
        max_file_count,
    )
    
    for count in tqdm(range(max_file_count)):
            
        input_dict = data_loader[count]
        text_inputs, image_inputs = input_dict["text_inputs"], input_dict["image_inputs"]
        save_path = f"{folder_path}/{input_dict['save_path']}.json"
            
        # two prompts
        prompts = {
            'detail': f"What is the {category_space['detail']} (of the main object) in the image based on the description? Answer from the following options: ",
            'obj': f"What is the main object in this image based on the description? Answer from the following options: ",
        }      
        
        checks, options, response, true_labels = {}, {}, {}, {}
        for mode in prompts:
            for i, item in enumerate(item_list[mode]):
                prompts[mode] += f" ({i+1}){item2word.get(item, item)}"
            prompts[mode] += ". Answer the number only and do not include any other texts (e.g., 1)."
            
            response[mode] = eval_llava(
                prompts[mode],
                [f"{root_dir}/datasets/action_bird/angry_bird.jpg"], # no images will be used, just put random image here to avoid errors
                llava_configs['tokenizer'],
                llava_configs['llava_model'],
                llava_configs['image_processor'],
                llava_configs['context_len'],
                llava_configs['llava_args'],
                device=llava_configs['device'],
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
    
            # for avoid unexpected error and retry the same prompt at most 10 times
            # normally it should not happen, but it happens for some models
            # such as gpt (network issue sometimes)
            retry = 0
            while retry <= 10:
                try:
                    out = call_model({
                        'text_inputs': text_inputs, 
                        'image_inputs': image_inputs,
                    })
                    break
                except:
                    retry += 1
                    print(f"retry {retry} times")
                    continue
            if retry > 10:
                print(f"retry {retry} times, skip")
                continue
            
            # save output
            save_json(out, save_path)
            print(f"save to {save_path}")
            
            # evaluate
            eval_llava(
                save_path,
                task_id,
                category_space,
                llava_configs,
            )
            print(f"evaluated")
    
    pass 

def eval_image(seed):
    set_seed(seed)
    pass

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description = 'Evaluate the results using LLaVA.')
    parser.add_argument('--task_id', type = int, default = 0, help = 'task id')
    parser.add_argument('--shot', type = int, default = 1, help = 'shot')
    parser.add_argument('--misleading', type = int, default = 0, help = 'misleading')
    parser.add_argument('--device', type = str, default = 'cuda', help = 'device')
    
    args = parser.parse_args()
    
    tokenizer, llava_model, image_processor, context_len, llava_args = load_llava(device = args.device)
    llava_configs = {
        'tokenizer': tokenizer,
        'llava_model': llava_model,
        'image_processor': image_processor,
        'context_len': context_len,
        'llava_args': llava_args,
        'device': args.device,
    }
