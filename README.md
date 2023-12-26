# Visual-ICL'

```
pip install tqdm pandas numpy random matplotlib scipy dashscope transformers
```

### google drive api
```
cd google_drive_helper
pip install google_auth_oauthlib
pip install --upgrade google-api-python-client
# add credentials.json to project_root
python google_download.py --user_email 'kangwj1995@furiosa.ai' --name gpt_evaluation_m
python google_download.py --user_email 'kangwj1995@furiosa.ai' --name gpt_evaluation
python google_download.py --user_email 'kangwj1995@furiosa.ai' --name llava_evaluation_m
python google_download.py --user_email 'kangwj1995@furiosa.ai' --name llava_evaluation
python google_download.py --user_email 'kangwj1995@furiosa.ai' --name clip_evaluation_m
python google_download.py --user_email 'kangwj1995@furiosa.ai' --name clip_evaluation
python google_download.py --user_email 'kangwj1995@furiosa.ai' --name exps
python google_download.py --name datasets --download_folder '.'
```

### Llava evaluation

```
cd models 
sudo apt-get update
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/liuhaotian/llava-v1.5-13b
```

### GPT-4V

```
call_gpt(
    text_inputs = ["Red", "Green", "Yellow"],
    mode = 'path', # 'url' or 'path'
    image_inputs = [
        "/skunk-pod-storage-yzeng58-40wisc-2eedu-pvc/micl/datasets/color object/color car/red car.jpg",
        "/skunk-pod-storage-yzeng58-40wisc-2eedu-pvc/micl/datasets/color object/color car/green car.jpg"
    ],
    max_tokens = 300,
    image_size = "1024x1024",
    quality = 'standard',
)
```

```
call_gpt(
    text_inputs = ["Red", "Green", "Yellow"],
    mode = 'url',
    image_inputs = [
        "https://media.istockphoto.com/id/1189903200/photo/red-generic-sedan-car-isolated-on-white-background-3d-illustration.jpg?s=612x612&w=0&k=20&c=uRu3o_h5FVljLQHS9z0oyz-XjXzzXN_YkyGXwhdMrjs=",
        "https://media.istockphoto.com/id/186872128/photo/a-bright-green-hatchback-family-car.jpg?s=2048x2048&w=is&k=20&c=vy3UZdiZFG_lV0Mp_Nka2DC4CglOqEuujpC-ra5TWJ0="
    ],
    max_tokens = 300,
    image_size = "1024x1024",
    quality = 'standard',
)
```