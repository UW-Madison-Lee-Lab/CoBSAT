task_types = [
    'color_object', 
    'weather_animal', 
    'style_object', 
    'action_animal', 
    'background_animal', 
    'texture_object',
]

item2word = {
    '3d': 'three-dimensional rendering',
    'park': 'amusement park',
}

item_dict = {
    'color': ['yellow', 'white', 'red', 'purple', 'pink', 'orange', 'green', 'brown', 'blue', 'black'],
    'object': ['leaf', 'hat', 'cup', 'chair', 'car', 'box', 'book', 'ball', 'bag', 'apple'],
    'weather': ['tornado', 'thunder', 'sunny', 'snowy', 'sandstorm', 'rainy', 'rainbow', 'hailstorm', 'foggy', 'aurora'],
    'animal': ['zebra', 'tiger', 'sheep', 'pig', 'monkey', 'lion', 'dog', 'cow', 'cat', 'bird'],
    'style': ['watercolor', 'sketch', 'pixel', 'origami', 'oil', 'lego', 'icon', 'graffiti', 'futuristic', '3d'],
    'action': ['swim', 'sleep', 'sing', 'run', 'read', 'fly', 'eat', 'drink', 'cry', 'angry'],
    'background': ['beach', 'desert', 'glacier', 'volcano', 'park', 'gym', 'waterfall', 'space', 'cave', 'seafloor'],
    'texture': ['wood', 'wicker', 'sequined', 'plastic', 'paper', 'metal', 'leather', 'lace', 'denim', 'ceramic'],
}

task_dataframe = {
    1: {
        'task_type': 'color_object',
        'x_space': 'color',
        'theta_space': 'object',
        'x_list': item_dict['color'],
        'theta_list': item_dict['object'],
    },
    2: {
        'task_type': 'color_object',
        'x_space': 'object',
        'theta_space': 'color',
        'x_list': item_dict['object'],
        'theta_list': item_dict['color'],
    },
    3: {
        'task_type': 'weather_animal',
        'x_space': 'weather',
        'theta_space': 'animal',
        'x_list': item_dict['weather'],
        'theta_list': item_dict['animal'],
    },
    4: {
        'task_type': 'weather_animal',
        'x_space': 'animal',
        'theta_space': 'weather',
        'x_list': item_dict['animal'],
        'theta_list': item_dict['weather'],
    },
    5: {
        'task_type': 'style_object',
        'x_space': 'style',
        'theta_space': 'object',
        'x_list': item_dict['style'],
        'theta_list': item_dict['object'],
    },
    6: {
        'task_type': 'style_object',
        'x_space': 'object',
        'theta_space': 'style',
        'x_list':  item_dict['object'],
        'theta_list': item_dict['style'],
    },
    7: {
        'task_type': 'action_animal',
        'x_space': 'action',
        'theta_space': 'animal',
        'x_list': item_dict['action'],
        'theta_list': item_dict['animal'],
    },
    8: {
        'task_type': 'action_animal',
        'x_space': 'animal',
        'theta_space': 'action',
        'x_list': item_dict['animal'],
        'theta_list': item_dict['action'],
    },
    9: {
        'task_type': 'background_animal',
        'x_space': 'background',
        'theta_space': 'object',
        'x_list': item_dict['background'],
        'theta_list': item_dict['animal'],
    },
    10: {
        'task_type': 'background_object',
        'x_space': 'object',
        'theta_space': 'background',
        'x_list': item_dict['animal'],
        'theta_list': item_dict['background'],
    },
    11: {
        'task_type': 'texture_object',
        'x_space': 'texture',
        'theta_space': 'object',
        'x_list': item_dict['texture'],
        'theta_list': item_dict['object'],
    },
    12: {
        'task_type': 'texture_object',
        'x_space': 'object',
        'theta_space': 'texture',
        'x_list': item_dict['object'],
        'theta_list': item_dict['texture'],
    },
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