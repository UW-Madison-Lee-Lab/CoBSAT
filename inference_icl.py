import os, argparse
from load_model import load_model
root_dir = os.path.dirname(os.path.abspath(__file__))

from helper import save_json, read_json, set_seed
from load_dataset import load_dataset
from environment import TRANSFORMER_CACHE
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMER_CACHE
from configs import task_dataframe, supported_models

prompts_list = read_json(f"{root_dir}/load_datasets/prompts_list.json")

def inference(
    model,
    call_model,
    shot,
    misleading,
    task_id,
    overwrite,
    gen_mode,
    max_file_count,
):
    misleading_flag = "_m" if misleading else ""
    base_path = f"{root_dir}/results/exps/{model}_{gen_mode}/shot_{shot}{misleading_flag}"
    
    folder_path = f"{base_path}/task_{task_id}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    data_loader = load_dataset(
        shot,
        misleading,
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
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                retry += 1
                print(f"Exception occurred: {type(e).__name__}, {e.args}")
                print('Retrying...')
                
        if retry > 10:
            out = {'description': 'ERROR', 'image': None, 'time': 0}
            print('ERROR')
            
        if gen_mode == 'text':
            save_json(out, save_path+'.json')
            print('---')
            print(out["description"])
        elif gen_mode == 'image':
            img = out['image']
            if img != None: img.save(save_path+'.jpg')
            
            out.pop('image')
            save_json(out, save_path+'.json')

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Generate images or image descriptions')
    parser.add_argument('--shot', type=int, nargs='+', default=[2,4,6,8])
    parser.add_argument('--misleading', type=int, nargs='+', default=[0,1], choices=[0,1])
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
        for misleading in args.misleading:
            for task_id in args.task_id:
                inference(
                    args.model,
                    call_model,
                    shot,
                    misleading,
                    task_id,
                    args.overwrite,
                    args.gen_mode,
                    args.max_file_count,
                )
