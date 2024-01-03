import os, argparse, random, glob
from load_model import load_model
from configs import task_dataframe
root_dir = os.path.dirname(os.path.abspath(__file__))
from helper import save_json, find_image, write_log, read_json

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
    log_path = f"{root_dir}/logs/text_inference/{task_id}/{model}/shot_{shot}{misleading_flag}.log"
    base_path = f"{root_dir}/results/exps/{model}_prompt2/shot_{shot}{misleading_flag}"
    
    folder_path = f"{base_path}/task_{task_id}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    x_list = task_dataframe[task_id]["x_list"]
    theta_list = task_dataframe[task_id]["theta_list"]
    
    for count in range(max_file_count):
        x_m_list = [x_list[x_index] + " " + theta_list[theta_index] for x_index, theta_index in zip(prompts_list[count]["x_list"], prompts_list[count]["theta_list"])] if misleading else [x_list[x_index] for x_index in prompts_list[count]["x_list"]]
        theta = theta_list[prompts_list[count]["theta_list"][shot+1]]

        text_inputs, image_inputs = [], []
        save_path = f"{folder_path}/{count}_{theta}_"
        print("========")
        print(theta)
        print("--------")
        for i in range(shot+1):
            text_inputs.append(x_m_list[i])
            if i < shot:
                image_path_i = find_image(
                    root_dir, 
                    task_id, 
                    x_list[i], 
                    theta, 
                )
                
                if image_path_i is None:
                    error_message = f"{task_id} {x_list[i]} {theta} not found!\n"
                    print(error_message)
                    write_log(log_path, error_message)
                    break
                image_inputs.append(image_path_i)
            print(x_m_list[i])
            save_path = save_path + "_" + x_list[i]
        if image_path_i is None: continue
        print("========")

        save_path = save_path + ".json"
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
    parser.add_argument('--model', type=str, default="qwen", choices = ['qwen', 'llava', 'gpt4v', 'emu2'])
    parser.add_argument('--max_file_count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--task_id', type=int, nargs='+', default=range(1,11))
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1])

    args = parser.parse_args()

    random.seed(args.seed)
    max_file_count = args.max_file_count
    call_model = load_model(args.model, args.device)

    for shot in args.shot:
        print(f"| shot: {shot}")
        for misleading in args.misleading:
            print(f"| ---- misleading: {misleading}")
            for task_id in args.task_id:
                print(f"| -------- task_id: {task_id}")
                inference(
                    args.model,
                    call_model,
                    shot,
                    misleading,
                    task_id,
                    args.overwrite,
                )
