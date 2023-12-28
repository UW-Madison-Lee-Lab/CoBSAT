import os, sys, random, numpy as np, torch, argparse
root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)
from call_gpt import call_gpt3_completion
from typing import Literal, cast
from configs import task_dataframe
from google_drive_helper.google_upload import drive_upload
from tqdm import tqdm
from helper import save_json

def evaluate_one_output(
    text_output,
    ground_truth_description, 
    prompt_idx = 1
):
    response = call_gpt3_completion(
        text_output,
        ground_truth_description,
        prompt_idx = prompt_idx,
    )
    
    if 'yes' in response.lower():
        return 1
    elif 'no' in response.lower():
        return 0
    else:
        return response
        
def get_ground_truth(
    x,
    theta,
    data_id,
):
    plural_dict = {
        'apple': 'apples',
        'cat': 'cats',
        'chair': 'chairs',
        'cup': 'cups',
        'dog': 'dogs',
        'lion': 'lions',
        'person': 'people',
        'shampoo': 'shampoo',
    }
    
    if data_id == 1:
        ground_truth_dict = {
            'overall': f'{x} {theta}',
            'textual': f"{x} object",
            'visual': f"a {theta}",
        }
    elif data_id == 2:
        ground_truth_dict = {
            'overall': f'{theta} {x}',
            'textual': f"a {x}",
            'visual': f"{theta} object",
        }
    elif data_id == 3:
        ground_truth_dict = {
            'overall': f"{x} {theta}" if x == 'one' else f"{x} {plural_dict[theta]}",
            'textual': f"{x} objects" if x!= 'one' else f"{x} object",
            'visual': f"{plural_dict[theta]}", 
        }
    elif data_id == 4:
        ground_truth_dict = {
            'overall': f"{theta} {x}" if theta == 'one' else f"{theta} {plural_dict[x]}s",
            'textual': f"{x}/{plural_dict[x]}",
            'visual': f"{theta} objects",
        }
    elif data_id == 5:
        ground_truth_dict = {
            'overall': f"{x} painting of {theta}" if x != 'oil painting' else f"{x} of {theta}",
            'textual': f"{x} painting" if x != 'oil painting' else f"{x}",
            'visual': f"painting of {theta}",
        }
    elif data_id == 6:
        ground_truth_dict = {
            'overall': f"{theta} painting of {x}" if theta != 'oil painting' else f"{theta} of {x}",
            'textual': f"painting of {x}",
            'visual': f"{theta} painting" if theta != 'oil painting' else f"{theta}",
        }
    elif data_id == 7:
        ground_truth_dict = {
            'overall': f"{theta} that is {x}ing",
            'textual': f"an animal/human that is {x}ing",
            'visual': f"{theta}",
        }
    elif data_id == 8:
        ground_truth_dict = {
            'overall': f"{x} that is {theta}ing",
            'textual': f"{x}",
            'visual': f"an animal/human that is {theta}ing",
        }
    elif data_id == 9:
        ground_truth_dict = {
            'overall': f"{x} on/in {theta}",
            'textual': f"{x}",
            'visual': f"{theta}",
        }
    elif data_id == 10:
        ground_truth_dict = {
            'overall': f"{theta} on/in {x}",
            'textual': f"{x}",
            'visual': f"{theta}",
        }
    else:
        raise ValueError("The data_id must be between 1 and 10.")
    
    return ground_truth_dict

def summary(
    data_ids,
    mllms,
    shots, 
    misleading = 0,
    overwrite = 1,
    prompt = 2,
):
    if misleading:
        google_folder = '1p6KBgFeQs1muXVg6badTQJvwitK7ayQ3'
        google_detail_folder = '1p6KBgFeQs1muXVg6badTQJvwitK7ayQ3'
        folder = f"{root_dir}/results/gpt_evaluation_m"
    else:
        google_folder = '10m4m8G-qv4s-JUEP0h7Mo0MFdnHFNGJA'
        google_detail_folder = '1r3WTYpSqOPYyu_2MW1ilXs6IFlX5AcAQ'
        folder = f"{root_dir}/results/gpt_evaluation"
        
    for data_id in data_ids: 
        print(f'Processing data_id: {data_id}')
        task_name = task_dataframe[data_id]['task_name']
        x_space = task_dataframe[data_id]['x_space']
        theta_space = task_dataframe[data_id]['theta_space']
        x_list = task_dataframe[data_id]['x_list']
        theta_list = task_dataframe[data_id]['theta_list']
        for mllm in mllms:
            print(f'| - mllm: {mllm}')
            for shot in shots:
                print(f"| --- shot {shot}")
                samples_mean = []
                misleading_flag = '_m' if misleading else ''
                cur_path = f"/pvc/ceph-block-kangwj1995/wisconsin/{mllm}/results/shot_{shot}{misleading_flag}/task_{data_id}" if prompt == 1 else f"/pvc/ceph-block-kangwj1995/wisconsin/{mllm}/results_prompt_2/shot_{shot}{misleading_flag}/task_{data_id}"
                samples = []
                all_generated_files = os.listdir(cur_path)
            
                # set seed for all experiments
                random.seed(123)
                np.random.seed(123)
                torch.manual_seed(123)
                
                corr_tot = {}
                for mode in ['overall', 'textual', 'visual']:
                    corr_tot[mode] = 0
                for filename in tqdm(all_generated_files):
                    with open(f"{cur_path}/{filename}", 'r') as f:
                        text_output = f.read()
                    file_path = os.path.join(cur_path, filename)
                    file_name = os.path.splitext(filename)[0]
                    theta = file_name.split('_')[1]
                    x = file_name.split('_')[-1]                    

                    corr = {}
                    ground_truth_dict = get_ground_truth(x, theta, data_id)

                    for mode in ['overall', 'textual', 'visual']:
                        type_mode = cast(Literal['overall', 'textual', 'visual'], mode)
                        ground_truth = ground_truth_dict[mode]
                        corr[mode] = evaluate_one_output(
                            text_output,
                            ground_truth,
                            prompt_idx=1, 
                        )
                        
                        # storing the error cases
                        if isinstance(corr[mode], str):
                            log_path = f"{folder}/log/text_gen/{data_id}_{task_name}_{mllm}_{shot}shot_{theta}_gpt_eval_error.json"
                            error_message = {
                                'filename': filename,
                                'text_output': text_output,
                                'ground_truth': ground_truth,
                                'mode': mode,
                            }
                            save_json(error_message, log_path)
                            corr[mode] = 0
                        print(corr[mode])
                        corr_tot[mode] += corr[mode]
                        
                    sample = {
                        'x': x,
                        'theta': theta,
                        'task_name': task_name,
                        'data_id': data_id,
                        'shot': shot,
                        'mllm': mllm,
                        'filename': filename,
                        'misleading': misleading,
                        'ground_truth': ground_truth_dict,
                        'text_output': text_output,
                        **corr,
                    }
                    samples.append(sample)
                json_file = f"{folder}/detail/[{data_id}]{task_name}_{mllm}_{shot}shot.json"
                save_json(samples, json_file)
                drive_upload([{
                        'mime_type': 'application/json',
                        'path': json_file
                    }], 
                    upload_folder=google_detail_folder,
                    overwrite=overwrite,
                )
                
                print(corr_tot)
                sample_mean = {}
                for mode in ['overall', 'textual', 'visual']:
                    sample_mean[mode] = corr_tot[mode]/len(all_generated_files)
                print(sample_mean)
                samples_mean.append(sample_mean)
                print(samples_mean)
                json_file = f"{folder}/[{data_id}]{task_name}_{mllm}_{shot}shot_mean.json"
                save_json(samples_mean, json_file)
                drive_upload([{
                        'mime_type': 'application/json',
                        'path': json_file
                    }], 
                    upload_folder=google_folder,
                    overwrite=overwrite,
                )
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIP evaluation')
    parser.add_argument('--data_ids', type=int, nargs='+', default=[1,2,3,4,5,6,7,8,9,10], help='data id', choices = list(range(1,11)))
    parser.add_argument('--mllms', type=str, nargs='+', default=['Emu', 'gill'], help='mllms')
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 2, 4], help='shots')
    parser.add_argument('--misleading', type=int, default=0, help='whether to use misleading data', choices = [0,1])
    parser.add_argument('--overwrite', type=int, default=1, help='whether to overwrite the original results in google drive', choices = [0,1])
    parser.add_argument('--prompt', type=int, default=2, help='prompt', choices = [1,2])
    args = parser.parse_args()

    # print experiment configuration
    args_dict = vars(args)
    print('Experiment Setting:')
    for key, value in args_dict.items():
        print(f"| {key}: {value}")

    summary(
        data_ids = args.data_ids,
        mllms = args.mllms, 
        shots = args.shots, 
        misleading = args.misleading,
        overwrite = args.overwrite,
        prompt = args.prompt,
    )
                    
                    