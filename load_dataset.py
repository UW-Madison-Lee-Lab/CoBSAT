import os
root_dir = os.path.dirname(os.path.abspath(__file__))

from helper import set_seed
import random
from configs import task_dataframe

def load_dataset(
    shot,
    misleading,
    task_id, 
    num_prompt = 1000,
    seed = 123,
):
    set_seed(123)
    misleading_flag = "_m" if misleading else ""
    
    x_list = task_dataframe[task_id]["x_list"]
    theta_list = task_dataframe[task_id]["theta_list"]
    
    for i in range(num_prompt):
        demo_x = random.sample(x_list, shot+1)
        demo_theta = random.sample(theta_list, shot+1)
        
        x_m_list = [f"{x} {theta}" for x, theta in zip(demo_x, demo_theta)] if misleading else demo_x
        theta = theta_list[shot+1]