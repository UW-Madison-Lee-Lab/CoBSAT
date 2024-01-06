from models.Emu.Emu2.emu.diffusion import EmuVisualGeneration
import torch
from PIL import Image
from environment import EMU2_PATH, EMU2_GEN_PATH

import cv2
from diffusers import DiffusionPipeline
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

"""
pipe = EmuVisualGeneration.from_pretrained(
        '/pvc/ceph-block-kangwj1995/wisconsin/copy2/Emu2-Gen/safety_checker/model.bf16.safetensors',
        dtype=torch.bfloat16,
        use_safetensors=True,
)

# Single GPU, e.g. cuda:0
pipe = pipe.multito(["cuda:0"])
# Multi GPU, e.g. cuda:0 and cuda:1
#pipe = pipe.multito(["cuda:0", "cuda:1"])
"""
#device = ["cuda:0"]

try:
    # For the non-first time of using, you can init the pipeline directly
    #print(sss)
    pipe = DiffusionPipeline.from_pretrained(
        EMU2_GEN_PATH,
        custom_pipeline="pipeline_emu2_gen",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="bf16",
    )#.eval()
except:
    # For the first time of using,
    # you need to download the huggingface repo "BAAI/Emu2-GEN" to local first
    
    multimodal_encoder = AutoModelForCausalLM.from_pretrained(
        f"{EMU2_GEN_PATH}/multimodal_encoder",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="bf16"
    )
    
    """
    multimodal_encoder = AutoModelForCausalLM.from_pretrained(
        f"{EMU2_GEN_PATH}/multimodal_encoder",
        load_in_4bit=True,
        trust_remote_code=True, 
        bnb_4bit_compute_dtype=torch.float16)
    """
    tokenizer = AutoTokenizer.from_pretrained(f"{EMU2_GEN_PATH}/tokenizer")
    
    pipe = DiffusionPipeline.from_pretrained(
        EMU2_GEN_PATH,
        custom_pipeline="pipeline_emu2_gen",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="bf16",
        multimodal_encoder=multimodal_encoder,
        tokenizer=tokenizer,
    )


pipe.to("cuda:0")





# text-to-image
prompt = "impressionist painting of an astronaut in a jungle"
ret = pipe(prompt)
ret.image.save("astronaut.png")

# image editing
image = Image.open("./models/Emu/Emu2/examples/dog.jpg").convert("RGB")
prompt = [image, "wearing a red hat on the beach.",image, "wearing a red hat on the beach."]
ret = pipe(prompt)
ret.image.save("dog_hat_beach.png")

# grounding generation
def draw_box(left, top, right, bottom):
    mask = np.zeros((448, 448, 3), dtype=np.uint8)
    mask = cv2.rectangle(mask, (left, top), (right, bottom), (255, 255, 255), 3)
    mask = Image.fromarray(mask)
    return mask

dog1 = Image.open("./models/Emu/Emu2/examples/dog1.jpg").convert("RGB")
dog2 = Image.open("./models/Emu/Emu2/examples/dog2.jpg").convert("RGB")
dog3 = Image.open("./models/Emu/Emu2/examples/dog3.jpg").convert("RGB")
dog1_mask = draw_box( 22,  14, 224, 224)
dog2_mask = draw_box(224,  10, 448, 224)
dog3_mask = draw_box(120, 264, 320, 438)

prompt = [
    "<grounding>",
    "An oil painting of three dogs,",
    "<phrase>the first dog</phrase>"
    "<object>",
    dog1_mask,
    "</object>",
    dog1,
    "<phrase>the second dog</phrase>"
    "<object>",
    dog2_mask,
    "</object>",
    dog2,
    "<phrase>the third dog</phrase>"
    "<object>",
    dog3_mask,
    "</object>",
    dog3,
]
ret = pipe(prompt)
ret.image.save("three_dogs.png")

# Autoencoding
# to enable the autoencoding mode, you can only input exactly one image as prompt
# if you want the model to generate an image,
# please input extra empty text "" besides the image, e.g.
#   autoencoding mode: prompt = image or [image]
#   generation mode: prompt = ["", image] or [image, ""]
prompt = Image.open("./models/Emu/Emu2/examples/doodle.jpg").convert("RGB")
ret = pipe(prompt)
ret.image.save("doodle_ae.png")