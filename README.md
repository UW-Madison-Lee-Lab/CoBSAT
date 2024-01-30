

<h1 align="center"> <p>Can MLLMs Perform Multimodal In-Context Learning for Text-to-Image Generation?</p></h1>
<h4 align="center">
    <p>
      <a href="https://yzeng58.github.io/zyc_cv/" target="_blank">Yuchen Zeng</a><sup>*1</sup>, <a href="https://scholar.google.com/citations?user=Q-ARWkwAAAAJ&hl=eh" target="_blank">Wonjun Kang*</a><sup>*2</sup>, <a href="https://bryce-chen.github.io/" target="_blank">Yicong Chen</a><sup>1</sup>, <a href="http://cvml.ajou.ac.kr/wiki/index.php/Professor" target="_blank">Hyung Il Koo</a><sup>2</sup>, <a href="https://kangwooklee.com/aboutme/" target="_blank">Kangwook Lee</a><sup>1</sup>
  </p>
  <p>
    <sup>1</sup>UW-Madison, <sup>2</sup> FuriosaAI
   </p>
    </h4>

**Paper Link**: TBA

**Abstract**:TBA

# News  🚀

* []...

# Contents

- [Step 1: Set Up Environment](# Step 1: Set Up Environment)
- [Step 2: Download Dataset](#Step 2: Download Dataset)
- [Evaluation: Benchmarking MLLMs' M-ICL Text-to-Image Capability](# Evaluation: Benchmarking MLLMs' M-ICL Text-to-Image Capability)

# Step 1: Set Up Environment

To set up the environment for benchmarking MLLMs, please follow the following steps.

1. Clone this repository and rename it as `micl`

   ```bash
   git clone https://github.com/UW-Madison-Lee-Lab/micl-imggen
   mv micl-imggen micl
   ```

2. Install Packages

   ```bash
   cd conda_env
   
   # create the environment that works for most of the cases
   conda env create -f general_env.yml
   
   # create the environment for llava (used for evaluating the accuracy of the images) to work 
   conda env create -f llava_env.yml
   ```

   

3. [Optional] If you want to conduct experiemnts on the models we featured, config the environment and download necessary files. 

   

# Step 2: Download Dataset

To use our dataset, please follow the following steps. 



Please download the images and their corresponding descriptions of our dataset from this [link](https://drive.google.com/file/d/1Yk8mo_xD95GEcee3EsA1qtJTsx1p1DNJ/view?usp=sharing).

<img width="903" alt="image" src="dataset_overview.jpg">



## llava

```
cd models
git clone https://github.com/haotian-liu/LLaVA.git
mv LLaVA llava
cd llava
```

# gill
```
mkdir models
cd models
mkdir gill
cd gill 
wget https://github.com/kohjingyu/gill/raw/main/checkpoints/gill_opt/decision_model.pth.tar
wget https://github.com/kohjingyu/gill/raw/main/checkpoints/gill_opt/pretrained_ckpt.pth.tar
wget https://huggingface.co/spaces/jykoh/gill/raw/main/gill/layers.py
wget https://huggingface.co/spaces/jykoh/gill/raw/main/gill/models.py
wget https://huggingface.co/spaces/jykoh/gill/raw/main/gill/utils.py
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
python google_download.py --name llava_evaluation_m/detail
python google_download.py --name llava_evaluation/detail
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