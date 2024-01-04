import os 
root_dir = os.path.dirname(os.path.abspath(__file__))

def load_model(
    model, 
    device = 'cuda',
    gen_mode = 'text',
):
    """
    Load models. 
    
    Examples:
    =========
    >>> from load_models import load_model
    >>> model = load_model('gpt4v')
    >>> call_model({
            'text_inputs': ["aurora", 'foggy', 'rainy'],
            'image_inputs': [
                "/data/yzeng58/micl/datasets/weather_pig/aurora_pig.jpg",
                "/data/yzeng58/micl/datasets/weather_pig/foggy_pig.jpg",
            ],
        })
    
    {'description': 'a pig in a city street at night with neon lights', 'gpt4v_time': 3.1531593799591064}
    
    >>> model = load_model('qwen')
    >>> call_model({
            'text_inputs': ["aurora", 'foggy', 'rainy'],
            'image_inputs': [
                "/data/yzeng58/micl/datasets/weather_pig/aurora_pig.jpg",
                "/data/yzeng58/micl/datasets/weather_pig/foggy_pig.jpg",
            ],
        })
        
    {'description': 'Picture 3:  A pig standing in the rain.',
    'history': [("I will provide you a few examples with text and image. Complete the example with the description of next image. Tell me only the text prompt and I'll use your entire answer as a direct input to A Dalle-3. Never say other explanations. auroraPicture 1: <img>/data/yzeng58/micl/datasets/weather_pig/aurora_pig.jpg</img>\nfoggyPicture 2: <img>/data/yzeng58/micl/datasets/weather_pig/foggy_pig.jpg</img>\nrainy",
    'Picture 3:  A pig standing in the rain.')],
    'time': 0.7774465084075928}
        
    >>> model = load_model('llava')
    >>> call_model({
            'text_inputs': ["aurora", 'foggy', 'rainy'],
            'image_inputs': [
                "/data/yzeng58/micl/datasets/weather_pig/aurora_pig.jpg",
                "/data/yzeng58/micl/datasets/weather_pig/foggy_pig.jpg",
            ],
        })
        
    {'description': 'The image features a pink pig walking down a dirt road in a foggy, misty environment. The pig is the main subject of the scene, and it appears to be the only animal present. The foggy atmosphere creates a sense of mystery and tranquility, as the pig seems to be exploring the area on its own. The dirt road stretches into the distance, leading the viewer to wonder what lies beyond the fog.',
    'time': 4.876652240753174}
        
        
    >>> model = load_model('gill')
    >>> call_model({
            'text_inputs': ["aurora", 'foggy', 'rainy'],
            'image_inputs': [
                "/data/yzeng58/micl/datasets/weather_pig/aurora_pig.jpg",
                "/data/yzeng58/micl/datasets/weather_pig/foggy_pig.jpg",
            ],
        })
        
    {'description': [' [IMG0] [IMG1] [IMG2] [IMG3] [IMG4] [IMG5] [IMG6] [IMG7]',
    {'gen': [(<PIL.Image.Image image mode=RGB size=512x512>, 0)],
    'ret': [],
    'decision': ['gen', [0, 1]]}],
    'time': 1.8119871616363525}
        
    """
    if model == 'gpt4v':
        if gen_mode != 'text':
            raise ValueError(f"gen_mode {gen_mode} not supported for gpt4v.")
        
        from load_models.call_gpt import call_gpt4v
        return lambda configs: call_gpt4v(
            mode = 'path', 
            use_dalle = False, 
            **configs,
        )
    elif model == 'qwen':
        if gen_mode != 'text':
            raise ValueError(f"gen_mode {gen_mode} not supported for qwen.")
        
        from load_models.call_qwen import load_qwen, call_qwen
        # sometimes there are some weird errors
        while True:
            try:
                model, tokenizer = load_qwen(device)
                call_qwen(model, tokenizer)
                return lambda configs: call_qwen(
                    model, 
                    tokenizer, 
                    **configs
                )
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                print(e)
                continue
    elif model == 'llava':
        if gen_mode != 'text':
            raise ValueError(f"gen_mode {gen_mode} not supported for llava.")
        
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
    elif model == 'gill':
        from load_models.call_gill import load_gill, call_gill
        while True:
            try:
                model, g_cuda = load_gill(device=device)
                call_gill(
                    model, 
                    g_cuda,
                    text_inputs = ["Red", "Green", "Yellow"],
                    image_inputs= [
                        f"{root_dir}/datasets/weather_pig/aurora_pig.jpg",
                        f"{root_dir}/datasets/weather_pig/hailstorm_pig.jpg"
                    ],
                    seed = 123,
                    gen_mode = gen_mode,
                )
                return lambda configs: call_gill(
                    model, 
                    g_cuda, 
                    gen_mode = gen_mode, 
                    **configs
                )
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                print(e)
                continue
    elif model == 'emu':
        from load_models.call_emu import load_emu, call_emu
        model = load_emu(device=device)
        call_emu(
            model, 
            text_inputs = ["Red", "Green", "Yellow"],
            image_inputs= [
                f"{root_dir}/datasets/weather_pig/aurora_pig.jpg",
                f"{root_dir}/datasets/weather_pig/hailstorm_pig.jpg"
            ],
            seed = 123,
            gen_mode = gen_mode,
            device = device,
        )
        return lambda configs: call_emu(
            model,
            gen_mode = gen_mode, 
            device = device,
            **configs
        )
    elif model == 'seed':
        from load_models.call_seed import load_seed, call_seed
        model, tokenizer, transform = load_seed(device=device)
        call_seed(
            model,
            tokenizer,
            transform, 
            text_inputs = ["Red", "Green", "Yellow"],
            image_inputs= [
                f"{root_dir}/datasets/weather_pig/aurora_pig.jpg",
                f"{root_dir}/datasets/weather_pig/hailstorm_pig.jpg"
            ],
            seed = 123,
            gen_mode = gen_mode,
             device = device,
        )
        return lambda configs: call_seed(
            model,
            tokenizer,
            transform,
            gen_mode = gen_mode,
            device = device, 
            **configs
        )
    else:
        raise ValueError(f"Model {model} not found.")

    