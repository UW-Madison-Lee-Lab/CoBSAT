import torch, os, json, argparse, sys, random, re
import numpy as np
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.eval.run_llava import eval_model
from llava.utils import disable_torch_init

# Configure the file saving
root_dir = os.path.dirname(os.getcwd())

sys.path.append(root_dir)
from configs import task_dataframe
sys.path.append(f"{root_dir}/google_drive_helper")
from google_upload import drive_upload

def save_json(data, path):
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

# Configure LlaVA
model_path = f"{root_dir}/evaluation/llava-v1.5-13b/"
# model_path = "/content/drive/MyDrive/Colab Notebooks/LLaVA-main/llava-v1.5-13b/"
model_name = "llava-v1.5-13b"
tokenizer, llava_model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name
)
print('LlaVA model loaded.')
llava_args = type('Args', (), {
    "conv_mode": None,
    "sep": ",",
    # "temperature": 0.2,
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()
disable_torch_init()

# Configure the LLaVA evaluation settings
"""
prompt_dict = {
    1: {
        'prompt1': 'What is the object\'s color in this image? Answer from the following options: (1)black (2)blue (3)green (4)pink (5)purple (6)red (7)white (8)yellow',
        'prompt2': 'What is the object in this image? Answer from the following options: (1)flower (2)leaf (3)box (4)building (5)cup (6)bag (7)chair (8)car',
    },
    3: {
        'prompt1': 'How many objects are there in this image? Answer from the following options: (1)one (2)two (3)three (4)four',
        'prompt2': 'What is the object in this image? Answer from the following options: (1)person (2)cat (3)apple (4)chair (5)cup (6)dog (7)lion (8)shampoo',
    },
    5: {
        'prompt1': 'What is the style of this image? Answer from the following options: (1)cartoon (2)oil (3)sketch (4)cubism (5)watercolor (6)origami',
        'prompt2': 'What is the object in this image? Answer from the following options: (1)man (2)cat (3)flower (4)apple (5)dog (6)house (7)car (8)chair',
    },
    7: {
        'prompt1': 'What is the object\'s action in this image? Answer from the following options: (1)wink (2)run (3)sing (4)fly (5)sit (6)drink (7)sleep (8)eat',
        'prompt2': 'What is the object in this image? Answer from the following options: (1)dog (2)cat (3)pig (4)lion (5)bird (6)monkey (7)man (8)woman',
    },
    9: {
        'prompt1': 'What is the background in this image? Answer from the following options: (1)beach (2)street (3)park (4)forest (5)office (6)classroom (7)gym (8)library',
        'prompt2': 'What is the object in this image? Answer from the following options: (1)dog (2)cat (3)robot (4)car (5)chair (6)monkey (7)man (8)woman',
    },
}
"""

prompt_dict = {
    1: {
        'prompt1': 'What is the object\'s color in this image? Answer from the following options: (1)black (2)blue (3)red (4)yellow (5)green (6)purple (7)pink (8)white',
        'prompt2': 'What is the object in this image? Answer from the following options: (1)car (2)leaf (3)box (4)building (5)cup (6)bag (7)flower (8)chair',
    },
    3: {
        'prompt1': 'How many objects are there in this image? Answer from the following options: (1)one (2)two (3)three (4)four (5)five (6)six',
        'prompt2': 'What is the object in this image? Answer from the following options: (1)apple (2)cat (3)chair (4)cup (5)dog (6)lion (7)person (8)shampoo',
    },
    5: {
        'prompt1': 'What is the style of this image? Answer from the following options: (1)cartoon (2)oil (3)sketch (4)cubism (5)watercolor (6)origami',
        'prompt2': 'What is the object in this image? Answer from the following options: (1)man (2)cat (3)flower (4)apple (5)dog (6)house (7)car (8)chair',
    },
    7: {
        'prompt1': 'What is the object\'s action in this image? Answer from the following options: (1)wink (2)run (3)sing (4)fly (5)sit (6)drink (7)sleep (8)eat',
        'prompt2': 'What is the object in this image? Answer from the following options: (1)cat (2)dog (3)pig (4)lion (5)bird (6)monkey (7)man (8)woman',
    },
    9: {
        'prompt1': 'What is the background in this image? Answer from the following options: (1)beach (2)street (3)park (4)forest (5)office (6)classroom (7)gym (8)library',
        'prompt2': 'What is the object in this image? Answer from the following options: (1)car (2)cat (3)chair (4)dog (5)man (6)monkey (7)robot (8)woman',
    },
}

for task_id in [2,4,6,8,10]:
    prompt_dict[task_id] = {}
    prompt_dict[task_id]['prompt1'] = prompt_dict[task_id-1]['prompt2']
    prompt_dict[task_id]['prompt2'] = prompt_dict[task_id-1]['prompt1']

def summary(
    data_ids,
    mllms,
    shots,
    misleading = 0,
    overwrite = 1,
):
    if misleading:
        google_folder = '1i21WRLal2Bsi_2QIQdc1Vd427up7g8N6'
        google_detail_folder = '1i70Ulvf81Peqp_Ch6byT5sZjjj7Ws7xl'
        folder = f"{root_dir}/results/llava_evaluation_m"
    else:
        google_folder = '1OffDVBzfnXA51wn-iqylxKtGCiAv3CiQ'
        google_detail_folder = '1OfwrMRitYzRJFqjfG24762cCpP8SgnKs'
        folder = f"{root_dir}/results/llava_evaluation"

    for data_id in data_ids:
        print(f'Processing data_id: {data_id}')
        task_name = task_dataframe[data_id]['task_name']
        x_space = task_dataframe[data_id]['x_space']
        theta_space = task_dataframe[data_id]['theta_space']
        x_list = task_dataframe[data_id]['x_list']
        theta_list = task_dataframe[data_id]['theta_list']
        prompt1 = prompt_dict[data_id]['prompt1']
        prompt2 = prompt_dict[data_id]['prompt2']

        for mllm in mllms:
            print(f'| - mllm: {mllm}')
            for shot in shots:
                print(f"| --- shot {shot}")
                score_per_task = 0
                strict_score_per_task = 0
                number_per_task = 0
                samples_mean = []



                #for theta in theta_list: # one function
                #print(f"| ------ theta: {theta}")
                misleading_flag = '_m' if misleading else ''
                #cur_path = f"{root_dir}/results/{mllm}_results/shot_{shot}{misleading_flag}/{x_space}_{theta_space}/{x_space}_{theta}"
                cur_path = f"/pvc/ceph-block-kangwj1995/wisconsin/{mllm}/results/shot_{shot}{misleading_flag}/task_{data_id}"
                score_per_object = 0
                strict_score_per_object = 0
                number_per_object = 0
                samples = []
                all_generated_files = os.listdir(cur_path)

                # set seed for all experiments
                random.seed(123)
                np.random.seed(123)
                torch.manual_seed(123)

                for filename in tqdm(all_generated_files):
                    if filename.endswith(".jpg") or filename.endswith(".png"):
                        image_file = os.path.join(cur_path, filename)
                        file_name = os.path.splitext(filename)[0]
                        theta = file_name.split('_')[1]
                        x_q = file_name.split('_')[-1]
                        out1 = eval_model(prompt1, image_file, tokenizer, llava_model, image_processor, context_len, llava_args)
                        out2 = eval_model(prompt2, image_file, tokenizer, llava_model, image_processor, context_len, llava_args)
                        #correct_caption = file_name.split('_')[-2] + ' ' + theta
                        correct_caption = x_q + ' ' + theta
                        correct_first = x_list.index(x_q) + 1
                        correct_second = theta_list.index(theta) + 1
                        strict_score = 0
                        score = 0
                        first_number = re.search(r'\d', out1)
                        if first_number:
                            first_number = int(first_number.group(0))
                            if first_number == correct_first:
                                score += 0.5
                        else:
                            print('out1 error:', out1)
                            with open(f'{folder}/error.txt', 'a') as f:
                                f.write('out1 error:' + out1 + '(' + filename + ')' + '\n')
                        second_number = re.search(r'\d', out2)
                        if second_number:
                            second_number = int(second_number.group(0))
                            if second_number == correct_second:
                                score += 0.5
                        else:
                            print('out2 error:', out2)
                            with open(f'{folder}/error.txt', 'a') as f:
                                f.write('out2 error:' + out2 + '(' + filename + ')' + '\n')
                        if score >= 1:
                            strict_score += 1
                        sample = {
                            "name": filename,
                            "correct_caption": correct_caption,
                            "first_ans": out1,
                            "sec_ans": out2,
                            "correct_first": correct_first,
                            "correct_second": correct_second,
                            "score": score,
                            "strict_score": strict_score
                        }
                        print('[', number_per_object, ']', sample)
                        samples.append(sample)
                        score_per_object += score
                        strict_score_per_object += strict_score
                        number_per_object += 1
                sample_mean = {
                    "name": task_name,
                    "score": score_per_object/number_per_object,
                    "strict_score": strict_score_per_object/number_per_object,
                    "number": number_per_object
                }
                samples_mean.append(sample_mean)
                samples.append(sample_mean)
                score_per_task += score_per_object
                strict_score_per_task += strict_score_per_object
                number_per_task += number_per_object
                json_file = f"{folder}/detail/[{data_id}]{task_name}_{mllm}_{shot}shot({(score_per_object/number_per_object):2f})({(strict_score_per_object/number_per_object):.2f}).json"
                save_json(samples, json_file)
                drive_upload([{
                        'mime_type': 'application/json',
                        'path': json_file
                    }], 
                    upload_folder=google_detail_folder,
                    overwrite=overwrite,
                )
                sample_mean = {
                    "name": task_name,
                    "score": score_per_task/number_per_task,
                    "strict_score": strict_score_per_task/number_per_task,
                    "number": number_per_task
                }
                samples_mean.append(sample_mean)
                json_file = f"{folder}/[{data_id}]{task_name}_{mllm}_{shot}shot_mean({(score_per_task/number_per_task):.2f})({(strict_score_per_task/number_per_task):.2f}).json"
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
