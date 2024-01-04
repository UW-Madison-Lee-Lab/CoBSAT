import os, argparse, random
from load_model import load_model
root_dir = os.path.dirname(os.path.abspath(__file__))

from helper import save_json, read_json
from load_dataset import load_dataset
from environment import TRANSFORMER_CACHE
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMER_CACHE

prompts_list = read_json(f"{root_dir}/load_datasets/prompts_list.json")

def inference(
    model,
    call_model,
    shot,
    misleading,
    task_id,
    overwrite,
):
    misleading_flag = "_m" if misleading else ""
    base_path = f"{root_dir}/results/exps/{model}_prompt2/shot_{shot}{misleading_flag}"
    
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
        if not overwrite and os.path.exists(save_path):
            print('skip')
            continue

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
                print(e)
                print('Retrying...')
                
        if retry > 10:
            out = {'description': 'ERROR'}
        
        print(out["description"])
        save_json(out, save_path)

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Generate image descriptions')
    parser.add_argument('--shot', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--misleading', type=int, nargs='+', default=[0,1], choices=[0,1])
    parser.add_argument('--model', type=str, default="qwen", choices = ['qwen', 'llava', 'gpt4v', 'emu2', 'emu', 'seed'])
    parser.add_argument('--max_file_count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--task_id', type=int, nargs='+', default=range(1,11))
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1])

    args = parser.parse_args()

    random.seed(args.seed)
    max_file_count = args.max_file_count
    call_model = load_model(
        args.model, 
        args.device, 
        gen_mode='text',
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
                )
