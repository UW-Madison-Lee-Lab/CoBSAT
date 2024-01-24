import os, argparse, pandas as pd
from load_model import load_model
root_dir = os.path.dirname(os.path.abspath(__file__))

from helper import save_json, read_json, set_seed
from load_dataset import load_dataset
from environment import TRANSFORMER_CACHE
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMER_CACHE
from configs import task_dataframe, supported_models, prompt_type_options, instruction_dict

prompts_list = read_json(f"{root_dir}/load_datasets/prompts_list.json")

def get_instruction(
    prompt_type, 
    gen_mode,
    task_id, 
    model,
):
    if prompt_type == 'instruct':
        return [instruction_dict[prompt_type][gen_mode][task_id], '']
    elif prompt_type == 'caption':
        return [instruction_dict[prompt_type][gen_mode], '']
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
            folder = os.path.basename(os.path.dirname(image_path))
            if 'action' in folder:
                file_name = 'action_animal'
            elif 'background' in folder:
                file_name = 'background_animal'
            elif 'color' in folder:
                file_name = 'color_object'
            elif 'style' in folder:
                file_name = 'style_object'
            elif 'texture' in folder:
                file_name = 'texture_object'
            else:
                raise ValueError(f"Unknown folder: {folder}!")
            
            data_df = pd.read_csv(f'{root_dir}/datasets/{file_name}.csv')
            text_inputs.insert(
                2*i+1, 
                data_df[data_df['image']==os.path.basename(image_path)]['caption'].values[0] + ' '
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
                )
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

def infer_model(
    call_model,
    prompt_type,
    text_inputs,
    image_inputs,
    task_id,
    model,
    gen_mode,
):  
    if prompt_type == 'cot':
        query = get_prompt(
            text_inputs,
            image_inputs,
            prompt_type,
            task_id, 
            model,
            'general', 
        )
        out = call_model(query)
        print('-------------------')
        print("CoT step:")
        print(f"{out['description']}\n")
        
        query = get_prompt(
            [],
            [],
            prompt_type,
            task_id, 
            model,
            gen_mode, 
            history = out['history'],
        )
        query['instruction'][1] = query['instruction'][1] + f"'{text_inputs[-1]}'."
        out = call_model(query)
    else:
        query = get_prompt(
            text_inputs,
            image_inputs,
            prompt_type,
            task_id, 
            model,
            gen_mode, 
        )
        out = call_model(query)
    return out

def inference(
    model,
    call_model,
    shot,
    prompt_type,
    task_id,
    overwrite,
    gen_mode,
    max_file_count,
):
    
    base_path = f"{root_dir}/results/exps/{model}_{gen_mode}/shot_{shot}/{prompt_type}"
    
    folder_path = f"{base_path}/task_{task_id}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    data_loader = load_dataset(
        shot,
        prompt_type,
        task_id,
        max_file_count,
    )
    
    for count in range(max_file_count):

        input_dict = data_loader[count]
        text_inputs, image_inputs = input_dict["text_inputs"], input_dict["image_inputs"]
        save_path = f"{folder_path}/{input_dict['save_path']}"
        
        print(f"===={count}-th sample====")
        print(f"theta: {input_dict['theta']}")
        for i in range(shot+1):
            print(f"{text_inputs[i]}")

        # skip if file exists
        if gen_mode == 'text':
            if not overwrite and os.path.exists(save_path+'.json'):
                print('skip')
                continue
        elif gen_mode == 'image':
            if not overwrite and os.path.exists(save_path+'.jpg'):
                print('skip')
                continue
        else:
            raise NotImplementedError(f"Unknown gen_mode: {gen_mode}!")
        
        out = infer_model(
            call_model,
            prompt_type,
            text_inputs,
            image_inputs,
            task_id,
            model,
            gen_mode,
        )
            
        out['text_inputs'] = text_inputs
        out['image_inputs'] = image_inputs
        if gen_mode == 'text':
            save_json(out, save_path+'.json')
            print('-------------------')
            print(out["description"])
        elif gen_mode == 'image':
            img = out['image']
            if img != None: img.save(save_path+'.jpg')
            print('-------------------')
            print(out["description"])
            out.pop('image')
            save_json(out, save_path+'.json')

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Generate images or image descriptions')
    parser.add_argument('--shot', type=int, nargs='+', default=[2,4,6,8])
    parser.add_argument('--prompt_type', type=str, nargs='+', default=['default'], choices=prompt_type_options)
    parser.add_argument('--model', type=str, default="qwen", choices = supported_models)
    parser.add_argument('--max_file_count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', nargs='+', type=str, default=['cuda']) # or ['35GiB', '25GiB', '35GiB']
    parser.add_argument('--task_id', type=int, nargs='+', default=list(task_dataframe.keys()))
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1])
    parser.add_argument('--gen_mode', type=str, default="image", choices=['text', 'image'])

    args = parser.parse_args()
    
    # print experiment configuration
    args_dict = vars(args)
    print("########"*3)
    print('## Experiment Setting:')
    print("########"*3)
    for key, value in args_dict.items():
        print(f"| {key}: {value}")
    
    if len(args.device) == 1: 
        device = args.device[0]
    else:
        device = {}
        for i in range(len(args.device)):
            device[i] = args.device[i]

    set_seed(args.seed)
    call_model = load_model(
        args.model, 
        device, 
        gen_mode=args.gen_mode,
    )

    for shot in args.shot:
        for prompt_type in args.prompt_type:
            for task_id in args.task_id:
                inference(
                    args.model,
                    call_model,
                    shot,
                    prompt_type,
                    task_id,
                    args.overwrite,
                    args.gen_mode,
                    args.max_file_count,
                )
