import os, argparse, random, glob
from load_model import load_model
from configs import google_folder_id, task_dataframe
root_dir = os.path.dirname(os.path.abspath(__file__))
from helper import save_json, find_image

def inference(
    model,
    call_model,
    shot,
    misleading,
    task_id,
):
    misleading_flag = "_m" if misleading else ""
    base_path = f"{root_dir}/results/exps/{model}_prompt2/shot_{shot}{misleading_flag}"
    
    
    folder_path = f"{base_path}/task_{task_id}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    x_list = task_dataframe[task_id]["x_list"]
    theta_list = task_dataframe[task_id]["theta_list"]
    
    count= 0
    while count < max_file_count:
        random.shuffle(x_list)
        random.shuffle(theta_list)
        x_m_list = [x + " " + theta for x, theta in zip(x_list, theta_list)] if misleading else x_list
        theta = theta_list[shot+1]

        text_inputs, image_inputs = [], []
        save_path = f"{folder_path}/{count}_{theta}_"
        print("========")
        print(theta)
        print("--------")
        for i in range(shot+1):
            text_inputs.append(x_m_list[i])
            image_path_i = find_image(
                root_dir, 
                task_id, 
                x_list[i], 
                theta, 
            )
            if i < shot:
                image_inputs.append(image_path_i)
            print(x_m_list[i])
            save_path = save_path + "_" + x_list[i]
        print("========")

        if count < len(glob.glob(folder_path + '/*.json')):
            #print("exist")
            count = count + 1
            continue
        elif count == len(glob.glob(folder_path + '/*.json')):
            #print("equal")
            count = count + 1
        
        while True:
            try:
                out = call_model({
                    'text_inputs': text_inputs, 
                    'image_inputs': image_inputs,
                })
                break
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                print(e)
                print('Retrying...')
        
        save_path = save_path + ".json"
        print(out["description"])

        save_json(out, save_path)

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Generate image descriptions')
    parser.add_argument('--shot', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--misleading', type=int, nargs='+', default=[0,1])
    parser.add_argument('--model', type=str, default="qwen")
    parser.add_argument('--max_file_count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--task_id', type=int, nargs='+', default=range(1,11))

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
                )