import pandas as pd

task_dataframe = pd.DataFrame({
    1: {
        'task_type': 'color_object',
        'x_space': 'color',
        'theta_space': 'object',
        'x_list': ['black', 'blue', 'red', 'yellow', 'green', 'purple', 'pink', 'white'],
        'theta_list': ['car', 'leaf', 'box', 'building', 'cup', 'bag', 'flower', 'chair'],
        'task_name': 'colortocolorobject',
    },
    2: {
        'task_type': 'color_object',
        'x_space': 'object',
        'theta_space': 'color',
        'x_list': ['car', 'leaf', 'box', 'building', 'cup', 'bag', 'flower', 'chair'],
        'theta_list': ['black', 'blue', 'red', 'yellow', 'green', 'purple', 'pink', 'white'],
        'task_name': 'objecttocolorobject',
    },
    3: {
        'task_type': 'count_object',
        'x_space': 'count',
        'theta_space': 'object',
        'x_list': ['one', 'two', 'three', 'four', 'five', 'six'],
        'theta_list': ['man', 'cat', 'flower', 'apple', 'dog', 'house', 'car', 'chair'],
        'task_name': 'counttocountobject',
    },
    4: {
        'task_type': 'count_object',
        'x_space': 'object',
        'theta_space': 'count',
        'x_list': ['man', 'cat', 'flower', 'apple', 'dog', 'house', 'car', 'chair'],
        'theta_list': ['one', 'two', 'three', 'four', 'five', 'six'],
        'task_name': 'objecttocountobject',
    },
    5: {
        'task_type': 'style_object',
        'x_space': 'style',
        'theta_space': 'object',
        'x_list': ['cartoon', 'oil painting', 'sketch', 'cubist', 'watercolor', 'origami'],
        'theta_list': ['man', 'cat', 'flower', 'apple', 'dog', 'house', 'car', 'chair'],
        'task_name': 'styletostyleobject',
    },
    6: {
        'task_type': 'style_object',
        'x_space': 'object',
        'theta_style': 'style',
        'x_list':  ['man', 'cat', 'flower', 'apple', 'dog', 'house', 'car', 'chair'],
        'theta_list': ['cartoon', 'oil painting', 'sketch', 'cubist', 'watercolor', 'origami'],
        'task_name': 'objecttostyleobject',
    },
    7: {
        'task_type': 'action_animal',
        'x_space': 'action',
        'theta_space': 'animal',
        'x_list': ['wink', 'run', 'sing', 'fly', 'sit', 'drink', 'sleep', 'eat'],
        'theta_list': ['cat', 'dog', 'pig', 'lion', 'bird', 'monkey', 'man', 'woman'],
        'task_name': 'actiontoactionanimal',
    },
    8: {
        'task_type': 'action_animal',
        'x_space': 'animal',
        'theta_space': 'action',
        'x_list': ['cat', 'dog', 'pig', 'lion', 'bird', 'monkey', 'man', 'woman'],
        'theta_list': ['wink', 'run', 'sing', 'fly', 'sit', 'drink', 'sleep', 'eat'],
        'task_name': 'animaltoactionanimal',
    },
    9: {
        'task_type': 'background_object',
        'x_space': 'background',
        'theta_space': 'object',
        'x_list': ['beach', 'street', 'park', 'forest', 'office', 'classroom', 'gym', 'library'],
        'theta_list': ['cat', 'dog', 'pig', 'lion', 'bird', 'monkey', 'man', 'woman'],
        'task_name': 'backgroundtobackgroundobject',
    },
    10: {
        'task_type': 'background_object',
        'x_space': 'object',
        'theta_space': 'background',
        'x_list': ['cat', 'dog', 'pig', 'lion', 'bird', 'monkey', 'man', 'woman'],
        'theta_list': ['beach', 'street', 'park', 'forest', 'office', 'classroom', 'gym', 'library'],
        'task_name': 'objecttobackgroundobject',
    }
})