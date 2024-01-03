import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from load_models.call_llava import load_llava
from models.llava.llava.eval.run_llava import eval_model
import argparse
from helper import set_seed

tokenizer, llava_model, image_processor, context_len, llava_args = load_llava(device='cuda')

def generate_single_caption(image_path):
    prompt = f"Generate a clear description of the image <image-placeholder>"
    caption = eval_model(
        prompt,
        [image_path],
        tokenizer,
        llava_model,
        image_processor,
        context_len,
        llava_args,
        device='cuda',
    )
    
    return caption

def generate_captions(
    dataset_folder_path,
    overwrite = False,
    test = False,
):
    # recursively find all images
    for root, dirs, files in os.walk(dataset_folder_path):
        for file in files:
            if file.endswith(".jpg"):
                
                image_path = os.path.join(root, file)
                caption_path = image_path.replace('datasets', 'results/captions').replace('.jpg', '.txt')
                folder = os.path.dirname(caption_path)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                    
                if os.path.exists(caption_path) and not overwrite:
                    continue
            
                caption = generate_single_caption(image_path)
                with open(caption_path, 'w') as f:
                    f.write(caption)
                    
                if test: break

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Generate image descriptions for the dataset')
    parser.add_argument('--overwrite', type=int, help='overwrite existing captions or not', choices = [0,1])
    parser.add_argument('--test', type=int, help='whether it is a test or not', choices = [0,1])
    
    args = parser.parse_args()
    
    set_seed(123)
    
    generate_captions(
        f"{root_dir}/datasets",
        overwrite = args.overwrite,
        test = args.test,
    )
        
    