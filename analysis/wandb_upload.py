import wandb, os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from configs import task_dataframe, supported_models, prompt_type_options
import pandas as pd, argparse
 
def update_wandb(
    model, 
    shot,
    prompt_type, 
    task_id,
    eval_mode, 
    finetuned,
    data_mode, 
    eval_mllm,
    ft_mode,
    eval_task_theme,
    seed = 123,
):
    task_type = task_dataframe[task_id]['task_type']
    category_space = {}
    category_space['detail'], category_space['obj'] = task_type.split('_')
    csv_file_path = f"{root_dir}/results/evals/{model}_{eval_mode}/shot_{shot}/{prompt_type}/task_{task_id}_summary.csv"
    
    # Initialize a new run
    wandb.init(
        project = 'micl',
        entity = 'lee-lab-uw-madison',
        config = {
            'task_id': task_id,
            'shot': shot,
            'prompt_type': prompt_type,
            'model': model,
            'seed': seed,
            'stage': 'eval',
            'file_type': eval_mode,
            'task_type': task_type,
            'x_space': task_dataframe[task_id]['x_space'],
            'theta_space': task_dataframe[task_id]['theta_space'],
            'finetuned': finetuned,
            'data_mode': data_mode,
            'eval_mllm': eval_mllm,
            'ft_mode': ft_mode,
            'eval_task_theme': eval_task_theme,
        },
    )
    
    df = pd.read_csv(csv_file_path)
    checks = {
        'detail': df.iloc[-1]['check_detail'],
        'obj': df.iloc[-1]['check_obj'],
        'textual': df.iloc[-1]['check_textual'],
        'visual': df.iloc[-1]['check_visual'], 
        'overall': df.iloc[-1]['correct'],
        'valid_count': 1000,
    }
    
    wandb.log(checks)
    
    # Finish the run
    wandb.finish()
    
if '__main__' == __name__:
    parser = argparse.ArgumentParser(description = 'Evaluate the results using LLaVA.')
    parser.add_argument('--model', type = str, default = 'qwen', choices = supported_models, help = 'model')
    parser.add_argument('--task_id', type = int, nargs = '+', default = list(task_dataframe.keys()), help = 'task id')
    parser.add_argument('--shot', type = int, nargs = '+', default = [2,4,6,8], help = 'shot')
    parser.add_argument('--prompt_type', type = str, nargs = '+', default = ['default'], help = 'prompt_type', choices = prompt_type_options)
    parser.add_argument('--seed', type = int, default = 123, help = 'seed')
    parser.add_argument('--eval_mode', type = str, default = 'text', help = 'evaluation mode', choices = ['text', 'image'])

    args = parser.parse_args()
    for task_id in args.task_id:
        for shot in args.shot:
            for prompt_type in args.prompt_type:
                update_wandb(
                    args.model, 
                    shot,
                    prompt_type, 
                    task_id,
                    args.eval_mode, 
                    seed = 123,
                )