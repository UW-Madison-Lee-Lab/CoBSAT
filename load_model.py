import os 
root_dir = os.path.dirname(os.path.abspath(__file__))
from environment import OPENAI_API_KEY
    
def load_model(
    model, 
    device = 'cuda',
    gen_mode = 'text',
    finetuned = False,
    shot = 2, 
    prompt_type = 'default',
    api_key = 'yz',
    ft_mode = 'all',
    eval_task_theme = '',
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
    if finetuned: 
        if not model in ['qwen', 'seed']:
            raise ValueError(f"finetuned is only supported for {model}. Only qwen and seed is supported.")
        
    if model == 'gpt4v':
        if gen_mode != 'text':
            raise ValueError(f"gen_mode {gen_mode} not supported for gpt4v.")
        
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY[api_key]
        from load_models.call_gpt import call_gpt4v
        return lambda configs: call_gpt4v(
            image_mode = 'path', 
            use_dalle = False, 
            api_key = api_key,
            **configs,
        )
    
    elif model == 'claude':
        if gen_mode != 'text':
            raise ValueError(f"gen_mode {gen_mode} not supported for claude.")
        
        from load_models.call_claude import load_claude, call_claude
        client = load_claude(api_key)
        return lambda configs: call_claude(
            client = client,
            **configs,
        )
        
    elif model == 'qwen':
        if gen_mode != 'text':
            raise ValueError(f"gen_mode {gen_mode} not supported for qwen.")
        
        from load_models.call_qwen import load_qwen, call_qwen
        # sometimes there are some weird errors

        model, tokenizer = load_qwen(
            device,
            finetuned = finetuned,
            shot = shot, 
            gen_mode = gen_mode, 
            prompt_type = prompt_type,
            ft_mode = ft_mode,
            eval_task_theme = eval_task_theme,
        )
        call_qwen(model, tokenizer)
        return lambda configs: call_qwen(
            model, 
            tokenizer, 
            **configs
        )
            
    elif model == 'llava':
        if gen_mode != 'text':
            raise ValueError(f"gen_mode {gen_mode} not supported for llava.")
        
        from load_models.call_llava import load_llava, call_llava

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
    elif model == 'llava16':
        if gen_mode != 'text':
            raise ValueError(f"gen_mode {gen_mode} not supported for llava.")
        
        from load_models.call_llava16 import load_llava16, call_llava16
        tokenizer, llava_model, image_processor, context_len, llava_args = load_llava16(device=device)
        
        call_llava16(
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
        return lambda configs: call_llava16(
            tokenizer,
            llava_model,
            image_processor,
            context_len,
            llava_args,
            device = device,
            **configs,
        )
        
    elif model == 'emu2':
        from load_models.call_emu2 import load_emu2, call_emu2

        model, tokenizer = load_emu2(device=device, gen_mode = gen_mode)
        
        call_emu2(
            model, 
            tokenizer,
            ['Yellow', 'White', 'Black'],
            [
                f'{root_dir}/models/Emu/Emu2/examples/dog2.jpg',
                f'{root_dir}/models/Emu/Emu2/examples/dog3.jpg',
            ],
            gen_mode = gen_mode,
        )
        
        return lambda configs: call_emu2(
            model, 
            tokenizer, 
            gen_mode = gen_mode,
            **configs
        )

    elif model == 'gill':
        from load_models.call_gill import load_gill, call_gill

        model, g_cuda = load_gill(device=device)
        call_gill(
            model, 
            g_cuda,
            text_inputs = ['Yellow', 'White', 'Black'],
            image_inputs= [
                f"{root_dir}/models/Emu/Emu2/examples/dog2.jpg",
                f"{root_dir}/models/Emu/Emu2/examples/dog3.jpg"
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
            
    elif model == 'emu':
        from load_models.call_emu import load_emu, call_emu
        model = load_emu(device=device, gen_mode=gen_mode)
        call_emu(
            model, 
            text_inputs = ['Yellow', 'White', 'Black'],
            image_inputs= [
                f"{root_dir}/models/Emu/Emu2/examples/dog2.jpg",
                f"{root_dir}/models/Emu/Emu2/examples/dog3.jpg"
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
        model, tokenizer, transform = load_seed(
            device = device,
            finetuned = finetuned,
            shot = shot,
            gen_mode = gen_mode,
            prompt_type = prompt_type,
            ft_mode = ft_mode,
            eval_task_theme = eval_task_theme,
        )
        call_seed(
            model,
            tokenizer,
            transform, 
            text_inputs = ['Yellow', 'White', 'Black'],
            image_inputs= [
                f"{root_dir}/models/Emu/Emu2/examples/dog2.jpg",
                f"{root_dir}/models/Emu/Emu2/examples/dog3.jpg"
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
    elif model == 'gemini':
        from load_models.call_gemini import load_gemini, call_gemini
        model = load_gemini(prompt_type, api_key)
        return lambda configs: call_gemini(
            model,
            **configs
        )
    else:
        raise ValueError(f"Model {model} not found.")

    