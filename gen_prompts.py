import argparse, random, json
from helper import save_json

parser = argparse.ArgumentParser(description='Generate prompts')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--max_file_count', type=int, default=1000)
parser.add_argument('--list_length', type=int, nargs='+', default=10)

args = parser.parse_args()

seed = args.seed
max_file_count = args.max_file_count
list_length = args.list_length

random.seed(seed)

x_list = list(range(list_length))
theta_list = list(range(list_length))

prompts_list = []

for i in range(max_file_count):
    random.shuffle(x_list)
    print(x_list)
    random.shuffle(theta_list)
    prompt_dict = {}
    prompt_dict["x_list"] = x_list.copy()
    prompt_dict["theta_list"] = theta_list.copy()
    prompts_list.append(prompt_dict)

with open("prompts_list.json", 'w') as f:
    json.dump(prompts_list, f, indent=4)

