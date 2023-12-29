from PIL import Image 
import requests
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

tokenizer = AutoTokenizer.from_pretrained("BAAI/Emu2") # "BAAI/Emu2-Chat"

with init_empty_weights():
     model = AutoModelForCausalLM.from_pretrained(
        "BAAI/Emu2", # "BAAI/Emu2-Chat"
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True)  

device_map = infer_auto_device_map(model, max_memory={0:'38GiB',1:'38GiB',}, no_split_module_classes=['Block','LlamaDecoderLayer'])  
# input and output logits should be on same device
device_map["model.decoder.lm.lm_head"] = 0

model = load_checkpoint_and_dispatch(
    model, 
    '/data/yzeng58/.cache/huggingface/hub/models--BAAI--Emu2/snapshots/fa835ec101e52da5e081695107e1ddd3c7c4d88a',
    device_map=device_map).eval()

# `[<IMG_PLH>]` is the image placeholder which will be replaced by image embeddings. 
# the number of `[<IMG_PLH>]` should be equal to the number of input images

query = '[<IMG_PLH>]Describe the image in details:' 
image = Image.open("./examples/blue_black_1_top_left.jpg").convert('RGB')

inputs = model.build_input_ids(
    text=[query],
    tokenizer=tokenizer,
    image=[image]

)

with torch.no_grad():
     outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image=inputs["image"].to(torch.bfloat16),
        max_new_tokens=64,
        length_penalty=-1)

output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)