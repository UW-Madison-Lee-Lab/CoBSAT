from load_models.call_gpt import call_gpt4v
from load_models.call_qwen import load_qwen, call_qwen

def load_model(model, device = 'cuda'):
    if model == 'gpt4v':
        return lambda configs: call_gpt4v(mode = 'path', use_dalle = False, **configs)
    elif model == 'qwen':
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
    else:
        raise ValueError(f"Model {model} not found.")
    