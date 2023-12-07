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
google_folder = '1OrDj-2dcy4-QV0MRdHalBFA8GASnXDD2'
google_detail_folder = '1OvHcYugDiB8DK0eIkftNAtx_WVgQtcmu'

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
    data_id,
    mllms, 
    shots, 
    misleading = 0,
    overwrite = 1,
):
    random.seed(123)
    # Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(123)
    # Set `torch` pseudo-random generator at a fixed value
    torch.manual_seed(123)

    task_name = task_dataframe[data_id]['task_name']
    x_space = task_dataframe[data_id]['x_space']
    theta_space = task_dataframe[data_id]['theta_space']
    x_list = task_dataframe[data_id]['x_list']
    theta_list = task_dataframe[data_id]['theta_list']
    folder = f"{root_dir}/results/clip_evaluation"
    for mllm in mllms:
        for shot in shots:
            score_per_task = 0
            strict_score_per_task = 0
            similarity_per_task = 0
            similarity1_per_task = 0
            similarity2_per_task = 0
            number_per_task = 0
            samples_mean = []
            for theta in theta_list: # one function
                misleading_flag = '_m' if misleading else ''
                cur_path = f"{root_dir}/results/{mllm}_results/shot_{shot}{misleading_flag}/{x_space}_{theta_space}/{x_space}_{theta}"
                score_per_object = 0
                strict_score_per_object = 0
                similarity_per_object = 0
                similarity1_per_object = 0
                similarity2_per_object = 0
                number_per_object = 0
                samples = []
                all_generated_files = os.listdir(cur_path)
                for filename in tqdm(all_generated_files):
                    if filename.endswith(".jpg") or filename.endswith(".png"):
                        image_file = os.path.join(cur_path, filename)
                        file_name = os.path.splitext(filename)[0]
                        correct_caption = file_name.split('_')[-2] + ' ' + theta
                        correct_first = x_list.index(file_name.split('_')[-2]) + 1
                        correct_second = theta_list.index(theta) + 1
                        text_list = [file_name.split('_')[-2], theta, correct_caption]
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
                    "name": theta,
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
                json_file = f"{folder}/detail/[{data_id}]{task_name}_{mllm}_{shot}shot_{theta}({(similarity_per_object/number_per_object):.2f})({(similarity1_per_object/number_per_object):.2f})({(similarity2_per_object/number_per_object):.2f})({(score_per_object/number_per_object):.2f})({(strict_score_per_object/number_per_object):.2f}).json"
                with open(json_file, 'w') as f:
                    json.dump(samples, f, indent=4)
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
            with open(json_file, 'w') as f:
                json.dump(samples_mean, f, indent=4)
            drive_upload([{
                    'mime_type': 'application/json',
                    'path': json_file
                }], 
                upload_folder=google_folder,
                overwrite=overwrite,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIP evaluation')
    parser.add_argument('--data_id', type=int, default=1, help='data id', choices = list(range(1,11)))
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
        data_id = args.data_id,
        mllms = args.mllms, 
        shots = args.shots, 
        misleading = args.misleading,
        overwrite = args.overwrite,
    )
