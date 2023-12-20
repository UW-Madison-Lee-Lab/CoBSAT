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
        raise ValueError("The response must be either yes or no.")
    ## TODO: log errors
    
def get_ground_truth(
    x,
    theta,
    data_id,
    mode: Literal['overall', 'textual', 'visual'] = 'overall', 
):
    plural_dict = {
        'man': 'men',
        'cat': 'cats',
        'flower': 'flowers',
        'apple': 'apples',
        'dog': 'dogs',
        'house': 'houses',
        'car': 'cars',
        'chair': 'chairs',
    }
    
    ground_truth_dict = {
        1: {
            'overall': f'{x} {theta}',
            'textual': f"{x} object",
            'visual': f"a {theta}", # TODO: test
        },
        2: {
            'overall': f'{theta} {x}',
            'textual': f"a {x}",
            'visual': f"{theta} object", # TODO: test
        },
        3: {
            'overall': f"{x} {theta}" if x == 'one' else f"{x} {plural_dict[theta]}",
            'textual': f"{x} objects",
            'visual': f"{plural_dict[theta]}", 
        },
        4: {
            'overall': f"{theta} {x}" if theta == 'one' else f"{theta} {plural_dict[x]}s",
            'textual': f"{x}/{plural_dict[x]}",
            'visual': f"{theta} objects",
        },
        5: {
            'overall': f"{x} painting of {theta}" if x != 'oil painting' else f"{x} of {theta}",
            'textual': f"{x} painting" if x != 'oil painting' else f"{x}",
            'visual': f"painting of {theta}",
        },        
        6: {
            'overall': f"{theta} painting of {x}" if theta != 'oil painting' else f"{theta} of {x}",
            'textual': f"painting of {x}",
            'visual': f"{theta} painting" if theta != 'oil painting' else f"{theta}",
        },
        7: {
            'overall': f"{theta} is {x}ing",
            'textual': f"an animal/human that is {x}ing",
            'visual': f"{theta}",
        },
        8: {
            'overall': f"{x} is {theta}ing",
            'textual': f"{x}",
            'visual': f"an animal/human that is {theta}ing",
        },
        9: {
            'overall': f"{x} on/in {theta}",
            'textual': f"{x}",
            'visual': f"{theta}",
        },
        10: {
            'overall': f"{theta} on/in {x}",
            'textual': f"{x}",
            'visual': f"{theta}",
        },
    }
    
    return ground_truth_dict[data_id][mode]

def summary(
    data_ids,
    mllms,
    shots, 
    misleading = 0,
    overwrite = 1,
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
                xxx
                for theta in theta_list:
                    print(f"| ------ theta: {theta}")
                    misleading_flag = '_m' if misleading else ''
                    # Wonjun: Please update the path for getting the textual output
                    cur_path = f"{root_dir}/results/{mllm}_results/shot_{shot}{misleading_flag}/{x_space}_{theta_space}/{x_space}_{theta}"
                    xx
                    samples = []
                    all_generated_files = os.listdir(cur_path)
                
                    # set seed for all experiments
                    random.seed(123)
                    np.random.seed(123)
                    torch.manual_seed(123)
                    
                    corr_tot = {}
                    # Wonjun: Please update the way of getting the textual output
                    for filename in tqdm(all_generated_files):
                        with open(f"{cur_path}/{filename}", 'r') as f:
                            text_output = f.read()
                        x = filename.split('.')[0].split('_')[-1]
                        
                        corr = {}
                        for mode in ['overall', 'textual', 'visual']:
                            type_mode = cast(Literal['overall', 'textual', 'visual'], mode)
                            corr[mode] = evaluate_one_output(
                                text_output,
                                get_ground_truth(
                                    x,
                                    theta,
                                    data_id,
                                    mode = type_mode,
                                ),
                            )
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
                            **corr,
                        }
                        samples.append(sample)
                    json_file = f"{folder}/detail/[{data_id}]{task_name}_{mllm}_{shot}shot_{theta}.json"
                    save_json(samples, json_file)
                    drive_upload([{
                            'mime_type': 'application/json',
                            'path': json_file
                        }], 
                        upload_folder=google_detail_folder,
                        overwrite=overwrite,
                    )
                sample_mean = {
                    
                }
                samples_mean.append(sample_mean)
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
    parser.add_argument('--data_ids', type=int, nargs='+', default=[1, 2], help='data id', choices = list(range(1,11)))
    parser.add_argument('--mllms', type=str, nargs='+', default=['emu', 'gill'], help='mllms')
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 2], help='shots')
    parser.add_argument('--misleading', type=int, default=0, help='whether to use misleading data', choices = [0,1])
    parser.add_argument('--overwrite', type=int, default=1, help='whether to overwrite the original results in google drive', choices = [0,1])

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
    )
                    
                    