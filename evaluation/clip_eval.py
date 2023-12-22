from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch, os, json, argparse, sys, random
import numpy as np
from tqdm import tqdm

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
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

def get_logits(model,text_embeds,image_embeds):
    logit_scale = model.logit_scale.exp()
    logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
    logits_per_image = logits_per_text.t()

    return logits_per_text,logits_per_image

def evaluation(image_file,text_list):
    image = Image.open(image_file).convert("RGB")
    inputs = processor(text=text_list, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    _,logits_per_image = get_logits(model,outputs.text_embeds,outputs.image_embeds)
    return logits_per_image[0].detach().cpu().numpy()

def summary(
    data_ids,
    mllms, 
    shots, 
    misleading = 0,
    overwrite = 1,
):
    if misleading:
        google_folder = '1i0XrxIbG8vHn8AE_J-Vo6MhoAN3S2Vfy'
        google_detail_folder = '1i4Zo98LxdmjQING1gFMt_tqHCwD2HWUV'
        folder = f"{root_dir}/results/clip_evaluation_m"
    else:
        google_folder = '1OrDj-2dcy4-QV0MRdHalBFA8GASnXDD2'
        google_detail_folder = '1OvHcYugDiB8DK0eIkftNAtx_WVgQtcmu'
        folder = f"{root_dir}/results/clip_evaluation"

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
                score_per_task = 0
                strict_score_per_task = 0
                similarity_per_task = 0
                similarity1_per_task = 0
                similarity2_per_task = 0
                number_per_task = 0
                samples_mean = []



                #for theta in theta_list: # one function
                #print(f"| ------ theta: {theta}")
                misleading_flag = '_m' if misleading else ''
                #cur_path = f"{root_dir}/results/{mllm}_results/shot_{shot}{misleading_flag}/{x_space}_{theta_space}/{x_space}_{theta}"
                cur_path = f"/pvc/ceph-block-kangwj1995/wisconsin/{mllm}/results/shot_{shot}{misleading_flag}/task_{data_id}"
                score_per_object = 0
                strict_score_per_object = 0
                similarity_per_object = 0
                similarity1_per_object = 0
                similarity2_per_object = 0
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
                        #correct_caption = file_name.split('_')[-2] + ' ' + theta
                        correct_caption = x_q + ' ' + theta
                        correct_first = x_list.index(x_q) + 1
                        correct_second = theta_list.index(theta) + 1
                        text_list = [x_q, theta, correct_caption]
                        text_list.extend(x_list)
                        text_list.extend(theta_list)
                        output = evaluation(image_file, text_list)
                        strict_score = 0
                        score = 0
                        similarity1 = float(output[0])
                        similarity2 = float(output[1])
                        similarity = float(output[2])
                        # get x_list position from output
                        out1 = output[3:3+len(x_list)]
                        out2 = output[3+len(x_list):3+len(x_list)+len(theta_list)]
                        first_number = np.argmax(out1) + 1
                        secnond_number = np.argmax(out2) + 1
                        if first_number == correct_first:
                            score += 0.5
                        if secnond_number == correct_second:
                            score += 0.5
                        if score >= 1:
                            strict_score += 1
                        sample = {
                            "name": filename,
                            "correct_caption": correct_caption,
                            # "first_ans": out1.tolist(),
                            # "sec_ans": out2.tolist(),
                            "correct_first": correct_first,
                            "correct_second": correct_second,
                            "score": score,
                            "strict_score": strict_score,
                            "similarity": similarity,
                            "similarity1": similarity1,
                            "similarity2": similarity2
                        }
                        samples.append(sample)
                        score_per_object += score
                        strict_score_per_object += strict_score
                        similarity_per_object += similarity
                        similarity1_per_object += similarity1
                        similarity2_per_object += similarity2
                        number_per_object += 1
                
                
                
                
                sample_mean = {
                    "name": task_name,
                    "score": score_per_object/number_per_object,
                    "strict_score": strict_score_per_object/number_per_object,
                    "similarity": similarity_per_object/number_per_object,
                    "similarity1": similarity1_per_object/number_per_object,
                    "similarity2": similarity2_per_object/number_per_object,
                    "number": number_per_object
                }
                samples_mean.append(sample_mean)
                samples.append(sample_mean)
                score_per_task += score_per_object
                strict_score_per_task += strict_score_per_object
                similarity_per_task += similarity_per_object
                similarity1_per_task += similarity1_per_object
                similarity2_per_task += similarity2_per_object
                number_per_task += number_per_object
                json_file = f"{folder}/detail/[{data_id}]{task_name}_{mllm}_{shot}shot({(similarity_per_object/number_per_object):.2f})({(similarity1_per_object/number_per_object):.2f})({(similarity2_per_object/number_per_object):.2f})({(score_per_object/number_per_object):.2f})({(strict_score_per_object/number_per_object):.2f}).json"
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
                        "similarity": similarity_per_task/number_per_task,
                        "similarity1": similarity1_per_task/number_per_task,
                        "similarity2": similarity2_per_task/number_per_task,
                        "number": number_per_task
                }
                samples_mean.append(sample_mean)
                json_file = f"{folder}/[{data_id}]{task_name}_{mllm}_{shot}shot_mean({(similarity_per_task/number_per_task):.2f})({(similarity1_per_task/number_per_task):.2f})({(similarity2_per_task/number_per_task):.2f})({(score_per_task/number_per_task):.2f})({(strict_score_per_task/number_per_task):.2f}).json"
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
    parser.add_argument('--shots', type=int, nargs='+', default=[1,2,4], help='shots')
    parser.add_argument('--misleading', type=int, default=1, help='whether to use misleading data', choices = [0,1])
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
