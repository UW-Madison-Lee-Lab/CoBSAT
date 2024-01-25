import os, json, numpy as np, random, torch, transformers, functools, time, pandas as pd
root_dir = os.path.dirname(os.path.abspath(__file__))
from PIL import Image
from configs import task_dataframe

def save_json(data, path):
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
        
def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def write_log(log_path, text):
    folder = os.path.dirname(log_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(log_path, 'a') as f:
        f.write(text)
        
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    transformers.set_seed(seed)
    
def get_image(name):
    extensions = ['jpg', 'webp', 'jpeg', 'png', 'JPG', 'Jpeg']
    found_image = None
    for ext in extensions:
        try:
            image_path = name+f'.{ext}'
            found_image = Image.open(image_path).convert('RGB')
            break
        except FileNotFoundError:
            continue

    if found_image is None:
        print(f"No valid image found for {name} !")
    return found_image

def find_image(
    root_dir, 
    task_id, 
    x_idx, 
    theta, 
):
    find = False
    
    task_type = task_dataframe[task_id]['task_type']
    category_space = {}
    category_space['detail'], category_space['obj'] = task_type.split('_')
    item_info = {}
    
    if task_dataframe[task_id]['x_space'] in ['object', 'animal']:
        item_info['obj'] = task_dataframe[task_id]['x_list'][x_idx]
        item_info['detail'] = theta
    else:
        item_info['obj'] = theta
        item_info['detail'] = task_dataframe[task_id]['x_list'][x_idx]
        
    
    folder_path = f"{root_dir}/datasets/{category_space['detail']}_{item_info['obj']}"
    image_path_i = f"{folder_path}/{item_info['detail']}_{item_info['obj']}.jpg"
        
    if os.path.exists(image_path_i):
        find = True
            
    if not find: 
        print(f"{image_path_i} not found!")
        return None 
    else:
        return image_path_i

def retry_if_fail(func):
    @functools.wraps(func)
    def wrapper_retry(*args, **kwargs):
        retry = 0
        while retry <= 10:
            try:
                out = func(*args, **kwargs)
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                retry += 1
                time.sleep(2)
                print(f"Exception occurred: {type(e).__name__}, {e.args}")
                print(f"Retry {retry} times...")

        if retry > 10:
            out = {'description': 'ERROR', 'image': None, 'time': 0}
            print('ERROR')
        
        return out
    return wrapper_retry

def find_caption(
    image_path, 
): 
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
    caption = data_df[data_df['image']==os.path.basename(image_path)]['caption'].values[0]
    return caption

def get_result_path(
    finetuned,
    model,
    gen_mode,
    shot,
    prompt_type,
):
    exp_folder = 'ft' if finetuned else 'exps'
    base_path = f"{root_dir}/results/{exp_folder}/{model}_{gen_mode}/shot_{shot}/{prompt_type}"
    if finetuned: base_path += "/exps"
    return base_path

def get_ft_path(
    model,
    gen_mode,
    shot,
    prompt_type,
):
    # output_dir = f'{root_dir}/results/ft/{model}_{gen_mode}/shot_{shot}/{prompt_type}/model'
    output_dir = f"{model}_{gen_mode}/shot_{shot}_{prompt_type}"
    data_path = f'{root_dir}/results/ft/{model}_{gen_mode}/shot_{shot}/{prompt_type}/dataset_ft.json'
    return {
        'model': output_dir,
        'data': data_path,
    }