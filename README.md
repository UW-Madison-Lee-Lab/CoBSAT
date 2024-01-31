

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

Stay tuned for the updates!

# Contents

- [Step 1: Set Up Environment](#set-up-environment)
- [Step 2: Download Dataset](#download-dataset)
- [Step 3: Benchmark MLLMs' M-ICL Text-to-Image Capability](#benchmark-mllms)

# Set Up Environment

To set up the environment for benchmarking MLLMs, please follow the following steps. This works for linux. 

1. Clone this repository and rename it as `micl`

   ```bash
   git clone https://github.com/UW-Madison-Lee-Lab/micl-imggen
   mv micl-imggen micl
   ```

2. Install Packages 

   <details><summary> Linux </summary>

   ```bash
   cd conda_env
   
   # create the environment that works for most of the cases
   conda create -n micl python=3.8.18
   pip install torch==2.1.2 torchvision==0.16.2 
   pip install -r conda_env/default_requirements.txt
   
   # create the environment for llava (used for evaluating the accuracy of the images) to work 
   conda create -n llava python=3.10.13
   pip install conda_env/llava_requirements.txt
   ```

   </details>

   <details><summary> Mac </summary>

   </details>

   <details><summary> Windows </summary>

   </details>

3. [Optional] If you want to conduct experiments on the models we featured, config the environment and download necessary files. 

4. Create `environment.py` in the `micl` directory. Note that many variables need you to config except `root_dir` on your own

   ```bash
   # Configure the environment variables for the project
   
   import os
   root_dir = os.path.dirname(os.path.abspath(__file__))
   
   OPENAI_API_KEY = f'{your_openai_key}'# NEED UPDATE
   TRANSFORMER_CACHE = '/data/yzeng58/.cache/huggingface/hub' # NEED UPDATE
   SEED_PROJECT_ROOT = f'{root_dir}/models/SEED'
   EMU_IMAGE_PATH = '/data/yzeng58/micl/models/Emu/Emu1/model_weights/Emu/pretrain' # [optional] NEED UPDATE
   EMU_INSTRUCT_PATH = '/data/yzeng58/micl/models/Emu/Emu1/model_weights/Emu/Emu-instruct.pt' # [optional] NEED UPDATE
   EMU_TEXT_PATH = '/data/yzeng58/micl/models/Emu/Emu1/model_weights/Emu/Emu-pretrain.pt' # [optional] NEED UPDATE
   ```

5. 

## Directory Structure

```
.
├── ...          
├── datasets                # will download the dataset in the next step
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



# Download Dataset

To use our dataset, please follow the following steps. 



Please download the images and their corresponding descriptions of our dataset from this [link](https://drive.google.com/file/d/1Yk8mo_xD95GEcee3EsA1qtJTsx1p1DNJ/view?usp=sharing).

<img width="903" alt="image" src="dataset_overview.jpg">

# [Optional] Feature Your Own Model

1. input your own model folder `OwnModel/` in `models/`.
2. create python file `call_OwnModel.py` in `load_models/` to load your own model.
3. add your own model in `load_model.py`

<details><summary> <code>call_OwnModel.py</code> example </summary>


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
        "/data/yzeng58/micl/datasets/weather_pig/aurora_pig.jpg",
        "/data/yzeng58/micl/datasets/weather_pig/hailstorm_pig.jpg"
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

<details><summary> <code>load_model.py</code> example </summary>


```
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

# Benchmark MLLMs

* Stage 1: Image (Description Generation)

  ```
  python inference_icl.py \
  --model seed \              # [seed, gill, emu, gpt4v, llava, qwen]
  --prompt_type default \     # [default, caption, instruct, misleading, cot, exact]
  --gen_mode image \          # [image, text]
  ```

  

* Stage 2: Evaluation

  ```
  python evaluation_icl.py
  --model seed \              # [seed, gill, emu, gpt4v, llava, qwen]
  --prompt_type default \     # [default, caption, instruct, misleading, cot, exact]
  --eval_mode image \          # [image, text]
  ```

  



## Citation

Please cite our work if you use this code.