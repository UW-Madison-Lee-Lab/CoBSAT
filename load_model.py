import os 
root_dir = os.path.dirname(os.path.abspath(__file__))

def load_model(model, device = 'cuda'):
    if model == 'gpt4v':
        from load_models.call_gpt import call_gpt4v
        return lambda configs: call_gpt4v(mode = 'path', use_dalle = False, **configs)
    elif model == 'qwen':
        from load_models.call_qwen import load_qwen, call_qwen
        # sometimes there are some weird errors
        while True:
            try:
                model, tokenizer = load_qwen(device)
                call_qwen(model, tokenizer)
                return lambda configs: call_qwen(model, tokenizer, **configs)
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                print(e)
                continue
    elif model == 'llava':
        from load_models.call_llava import call_llava, load_llava
        while True:
            try:
                tokenizer, llava_model, image_processor, context_len, llava_args = load_llava(device=device)
                call_llava(
                    tokenizer,
                    llava_model,
                    image_processor,
                    context_len,
                    llava_args,
                    text_inputs = ["Red", "Green", "Yellow"],
                    image_inputs = [
                        "https://media.istockphoto.com/id/1189903200/photo/red-generic-sedan-car-isolated-on-white-background-3d-illustration.jpg?s=612x612&w=0&k=20&c=uRu3o_h5FVljLQHS9z0oyz-XjXzzXN_YkyGXwhdMrjs=",
                        "https://media.istockphoto.com/id/186872128/photo/a-bright-green-hatchback-family-car.jpg?s=2048x2048&w=is&k=20&c=vy3UZdiZFG_lV0Mp_Nka2DC4CglOqEuujpC-ra5TWJ0="
                    ],
                    seed = 123,
                    device = device,
                )
                return lambda configs: call_llava(
                    tokenizer,
                    llava_model,
                    image_processor,
                    context_len,
                    llava_args,
                    device = device,
                    **configs,
                )
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                print(e)
                continue
    elif model == 'emu2':
        from load_models.call_emu2 import load_emu2, call_emu2
        while True:
            try:
                model, tokenizer = load_emu2(None)
                call_emu2(
                    model, 
                    tokenizer,
                    ['Yellow', 'White', 'Black'],
                    [
                        f'{root_dir}/models/Emu/Emu2/examples/dog2.jpg',
                        f'{root_dir}/models/Emu/Emu2/examples/dog3.jpg',
                    ],
                )
                return lambda configs: call_emu2(model, tokenizer, **configs)
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                print(e)
                continue
    else:
        raise ValueError(f"Model {model} not found.")

    