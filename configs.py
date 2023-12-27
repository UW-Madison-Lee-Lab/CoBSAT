import pandas as pd

task_dataframe = {
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
        'theta_list': ['apple', 'cat', 'chair', 'cup', 'dog', 'lion', 'person', 'shampoo'],
        'task_name': 'counttocountobject',
    },
    4: {
        'task_type': 'count_object',
        'x_space': 'object',
        'theta_space': 'count',
        'x_list': ['apple', 'cat', 'chair', 'cup', 'dog', 'lion', 'person', 'shampoo'],
        'theta_list': ['one', 'two', 'three', 'four', 'five', 'six'],
        'task_name': 'objecttocountobject',
    },
    5: {
        'task_type': 'style_object',
        'x_space': 'style',
        'theta_space': 'object',
        'x_list': ['cartoon', 'oil', 'sketch', 'cubism', 'watercolor', 'origami'],
        'theta_list': ['man', 'cat', 'flower', 'apple', 'dog', 'house', 'car', 'chair'],
        'task_name': 'styletostyleobject',
    },
    6: {
        'task_type': 'style_object',
        'x_space': 'object',
        'theta_space': 'style',
        'x_list':  ['man', 'cat', 'flower', 'apple', 'dog', 'house', 'car', 'chair'],
        'theta_list': ['cartoon', 'oil', 'sketch', 'cubism', 'watercolor', 'origami'],
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
        'theta_list': ['car', 'cat', 'chair', 'dog', 'man', 'monkey', 'robot', 'woman'],
        'task_name': 'backgroundtobackgroundobject',
    },
    10: {
        'task_type': 'background_object',
        'x_space': 'object',
        'theta_space': 'background',
        'x_list': ['car', 'cat', 'chair', 'dog', 'man', 'monkey', 'robot', 'woman'],
        'theta_list': ['beach', 'street', 'park', 'forest', 'office', 'classroom', 'gym', 'library'],
        'task_name': 'objecttobackgroundobject',
    }
}

google_folder_id = {
    'llava_evaluation_m': '1i21WRLal2Bsi_2QIQdc1Vd427up7g8N6',
    'llava_evaluation_m/detail': '1i70Ulvf81Peqp_Ch6byT5sZjjj7Ws7xl',
    'llava_evaluation': '1OffDVBzfnXA51wn-iqylxKtGCiAv3CiQ',
    'llava_evaluation/detail': '1OfwrMRitYzRJFqjfG24762cCpP8SgnKs',
    'clip_evaluation_m': '1i0XrxIbG8vHn8AE_J-Vo6MhoAN3S2Vfy',
    'clip_evaluation_m/detail': '1i4Zo98LxdmjQING1gFMt_tqHCwD2HWUV',
    'clip_evaluation': '1OrDj-2dcy4-QV0MRdHalBFA8GASnXDD2',
    'clip_evaluation/detail': '1OvHcYugDiB8DK0eIkftNAtx_WVgQtcmu',
    'gpt_evaluation_m': '1p6KBgFeQs1muXVg6badTQJvwitK7ayQ3',
    'gpt_evaluation_m/detail': '1iGcGnInHx6uAwBkv91RgDIW69yi6EMSd',
    'gpt_evaluation': '10m4m8G-qv4s-JUEP0h7Mo0MFdnHFNGJA',
    'gpt_evaluation/detail': '1r3WTYpSqOPYyu_2MW1ilXs6IFlX5AcAQ',
    'exps': '1fEMxOiV4xKhNpVDSTiJCwvobROv9BiNb',
    'datasets': '1mCmuB1IPb6Ht_ZZLfPjip4DCh-cWzO9D',
}