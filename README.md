

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

* TODO: Our dataset will be available on huggingface! 

Stay tuned for the updates!

# Contents

- [Step 1: Set Up Environment](#set-up-environment)
- [Step 2: Download Dataset](#download-dataset)
- [Step 3: Benchmark MLLMs' M-ICL Text-to-Image Capability](#benchmark-mllms)

# Set Up Environment

To set up the environment for benchmarking MLLMs, please follow the following steps. This works for linux. 

1. Clone this repository and rename it as `cobsat`

   ```bash
   git clone --recurse-submodules https://github.com/UW-Madison-Lee-Lab/CoBSAT
   mv CoBSAT cobsat
   cd cobsat
   ```

2. Install Packages 

   <details><summary> Linux </summary>

   ```bash
   cd conda_env
   
   # create the environment that works for most of the cases
   conda create -n cobsat python=3.8.18
   conda activate cobsat
   pip install torch==2.1.2 torchvision==0.16.2 
   pip install -r conda_env/default_requirements.txt
   
   # create the environment for llava (used for evaluating the accuracy of the images) to work 
   conda create -n llava python=3.10.13
   cd models/llava
   conda activate llava
   pip install --upgrade pip  # enable PEP 660 support
   pip install -e .
   conda ../..
   pip install conda_env/llava_requirements.txt
   ```

   </details>

   <details><summary> Mac </summary>

   </details>

   <details><summary> Windows </summary>
   </details>

3. Create `environment.py` in the `cobsat` directory. Note that many variables need you to config except `root_dir` on your own

   ```python
   # Configure the environment variables for the project
   
   import os
   root_dir = os.path.dirname(os.path.abspath(__file__))
   
   SEED_PROJECT_ROOT = f'{root_dir}/models/SEED'
   
   ###############
   # NEED UPDATE #
   ###############
   TRANSFORMER_CACHE = '/data/yzeng58/.cache/huggingface/hub' 
   
   #########################
   # NEED UPDATE IF NEEDED #
   #########################
   # GPT-4V
   OPENAI_API_KEY = f'{your_openai_key}'
   # Emu for Image Generation
   EMU_IMAGE_PATH = '/data/yzeng58/cobsat/models/Emu/Emu1/model_weights/Emu/pretrain' 
   # Emu-Instruct
   EMU_INSTRUCT_PATH = '/data/yzeng58/cobsat/models/Emu/Emu1/model_weights/Emu/Emu-instruct.pt' 
   # Emu-Generation
   EMU_TEXT_PATH = '/data/yzeng58/cobsat/models/Emu/Emu1/model_weights/Emu/Emu-pretrain.pt'
   # WANDB Logging https://wandb.ai/site
   WANDB_ENTITY = 'lee-lab-uw-madison'
   WANDB_PROJECT = 'cobsat'
   ```

# Download Dataset

<img width="903" alt="image" src="dataset_overview.jpg">

To use our dataset, please follow the following steps. 

1. Download the images and their corresponding descriptions of our dataset from this [link](https://drive.google.com/file/d/1Yk8mo_xD95GEcee3EsA1qtJTsx1p1DNJ/view?usp=sharing).
2. Untar the `.tar.gz` file via `tar -xvf datasets.tar.gz` and extract the `datasets` folder into your `cobsat` folder. 

Up to now, the structure of your `cobsat` folder should look like this.

```
.
├── ...          
├── datasets                # download the dataset in this step
├── load_models
│   ├── call_emu.py
│		├── call_gill.py
│		├── call_gpt.py
│		├── call_llava.py
│		├── call_qwen.py
│		├── call_seed.py
│   ├── call_your_model.py  # [optional] create python file to load the model you want to evaluate
│   └── ... 
├── models                  
│   ├── SEED                # [optional] git clone https://github.com/AILab-CVC/SEED
│   ├── gill                # [optional] git clone https://github.com/kohjingyu/gill
│   ├── Emu                 # [optional] git clone https://github.com/baaivision/Emu
│   │   └── Emu1 
│   ├── LLaVA               # [optional] git clone https://github.com/haotian-liu/LLaVA
│   ├── Qwen-VL             # [optional] git clone https://github.com/QwenLM/Qwen-VL
│   ├── OwnModel            # [optional] input your own model folder
│   └── ...
├── ...
├── environment.py          # follow the instruction above to create this file
├── load_model.py           # [optional] add your own model                
└── ...
```

# Benchmark MLLMs

### Already Supported Models

- [x] [SEED-LLaMA](https://arxiv.org/abs/2310.01218)
  * Image Generation
  * Text Generation
- [x] [GILL](https://arxiv.org/abs/2305.17216)
  * Image Generation
  * Text Generation
- [x] [Emu](https://arxiv.org/abs/2307.05222)
  * Image Generation
  * Text Generation

- [x] [GPT-4V]()
  * Text Generation
- [x] [LLaVA](https://arxiv.org/abs/2304.08485)
  * Text Generation
- [x] [Qwen-VL](https://arxiv.org/abs/2308.12966)
  * Text Generation
  * Fine-Tuning

### [Optional] Feature Your Own Model

1. Create your own model folder `OwnModel/` in `models/` if needed.

2. Create python file `call_OwnModel.py` in `load_models/` to load your own model.

   <details><summary> <code>call_OwnModel.py</code> template </summary>

   ```
   def load_OwnModel(
       device = 'cuda',
       seed = 123,
   ):
   
       return model, others
   ```

   ```
   def call_OwnModel(
       model, 
       others,
       text_inputs = ["Red", "Green", "Yellow"],
       image_inputs = [
           "/data/yzeng58/cobsat/datasets/weather_pig/aurora_pig.jpg",
           "/data/yzeng58/cobsat/datasets/weather_pig/hailstorm_pig.jpg"
       ],
       seed = 123,
       gen_mode = 'text',
   ):
   
       output_dict = {}
       OwnModel_start = time()
   
       if gen_mode == 'image':
   
       elif gen_mode == 'text':
   
       OwnModel_end = time()
       output_dict['time'] = OwnModel_end - OwnModel_start
   
       return output_dict
   ```

   </details>

3. Add your own model in `load_model.py`.

   <details><summary> <code>load_model.py</code> template </summary>

   ```python
       elif model == 'OwnModel':
           from load_models.call_OwnModel import load_OwnModel, call_OwnModel
   
           model, others = load_OwnModel(device=device)
           call_OwnModel(
               model, 
               others,
               text_inputs = ['Yellow', 'White', 'Black'],
               image_inputs= [
                   f"{root_dir}/models/Emu/Emu2/examples/dog2.jpg",
                   f"{root_dir}/models/Emu/Emu2/examples/dog3.jpg"
               ],
               seed = 123,
               gen_mode = gen_mode,
           )
           return lambda configs: call_OwnModel(
               model, 
               others, 
               gen_mode = gen_mode, 
               **configs
           )
   ```

   </details>

4. Add your model to `supported_models` in `configs.py`.

   ```python
   supported_models = [
       'qwen', 
       'llava', 
       'gpt4v', 
       'emu2', 
       'emu', 
       'seed',
       'gill',
       'OwnModel', # your own model
   ]
   ```

5. Config the default instruction of your model by updating `instruction_dict` in `configs.py`.

   Especially, you need to edit the `instruction_dict['default']`.

   If your model support image generation, then you need to edit `instruction_dict['default']['image']` .

   ```python
   				'image': {
               'gill': (
                   'You are a professional assistant can generate a new image based on the seqeunce. ',
                   '',
               ),
   						...
             	# NEED UPDATE
         			'OwnModel': (
               		'Based on the sequence, generate the next image.',
                 	'Make the prediction now.'
               )
           }
   ```

   If your model support text generation, then you need to edit `instruction_dict['default']['text']` .

   ```python
   				'text': {
               'seed': (
                   "I will provide you a few examples with text and image. Complete the example with the description of next image. Tell me only the text prompt and I'll use your entire answer as a direct input to A Dalle-3. Never say other explanations. ",
                   '',
               ),
   						...
             	# NEED UPDATE
         			'OwnModel': (
               		'Based on the sequence, describe the next image clearly, including details such as the main object, color, texture, background, action, style, if applicable. ',
                 	'Make the prediction now.'
               )
           }
   ```

6. [Optional: If you want to finetune your model on our dataset] 

​	TBA

### Evaluate MLLMs on Our Dataset

* [Optional] Fine-tuning

  ```bash
  # Example
  python finetune_icl.py \
  --model qwen \ 							# model you want to finetune
  --shot 2 \ 		 							# number of demonstrations
  --prompt_type default \			# prompt type
  --gen_mode text							# [image, text]
  ```

* Stage 1: Output Generation

  ```bash
  python inference_icl.py \
  --model seed \              # [seed, gill, emu, gpt4v, llava, qwen]
  --prompt_type default \     # [default, caption, instruct, misleading, cot, exact]
  --gen_mode image \          # [image, text]
  --shot 2
  ...
  ```

* Stage 2: Evaluation

  ```bash
  python evaluation_icl.py
  --model seed \              # [seed, gill, emu, gpt4v, llava, qwen]
  --prompt_type default \     # [default, caption, instruct, misleading, cot, exact]
  --eval_mode image \          # [image, text]
  ...
  ```


### Description of Parameters

TBA

## Citation

TBA