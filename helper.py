import os, json, numpy as np, random, torch
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
        
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
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
    x, 
    theta, 
):
    image_path_i = None
    
    try:
        image_path_prefix = f"{root_dir}/datasets/{task_dataframe[task_id]['task_type']}/{x} {theta}"
            
        for file_type in ['jpg', 'webp', 'png', 'jpeg', 'JPG', 'Jpeg']:
            image_path_i = f"{image_path_prefix}.{file_type}"
            if os.path.exists(image_path_i):
                break
    except FileNotFoundError:
        image_path_prefix = f"{root_dir}/datasets/{task_dataframe[task_id]['task_type']}/{theta} {x}"

        for file_type in ['jpg', 'webp', 'png', 'jpeg', 'JPG', 'Jpeg']:
            image_path_i = f"{image_path_prefix}.{file_type}"
            if os.path.exists(image_path_i):
                break
    except KeyboardInterrupt:
        exit()
    except Exception as e:
        print(e)
        print(f"Error: {task_id} {x}, {theta}")
        
    return image_path_i