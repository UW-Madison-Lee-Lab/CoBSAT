task_types = [
    'color_object',  
    'style_object', 
    'action_animal', 
    'texture_object',
    'background_animal',
]

item2word = {
    '3d': 'wireframe',
    'park': 'amusement park',
}

item_dict = {
    'color': ['yellow', 'white', 'red', 'purple', 'pink', 'orange', 'green', 'brown', 'blue', 'black'],
    'object': ['leaf', 'hat', 'cup', 'chair', 'car', 'box', 'book', 'ball', 'bag', 'apple'],
    'weather': ['tornado', 'thunder', 'sunny', 'snowy', 'sandstorm', 'rainy', 'rainbow', 'hailstorm', 'foggy', 'aurora'],
    'animal': ['zebra', 'tiger', 'sheep', 'pig', 'monkey', 'lion', 'dog', 'cow', 'cat', 'bird'],
    'style': ['watercolor', 'sketch', 'pixel', 'origami', 'lego', 'icon', 'graffiti', 'futuristic', '3d', 'old'],
    'action': ['swim', 'sleep', 'sing', 'run', 'read', 'fly', 'eat', 'drink', 'cry', 'angry'],
    'background': ['beach', 'desert', 'glacier', 'volcano', 'park', 'gym', 'waterfall', 'space', 'cave', 'seafloor'],
    'texture': ['wood', 'wicker', 'sequined', 'plastic', 'paper', 'metal', 'leather', 'lace', 'denim', 'ceramic'],
}

task_dataframe = {
    1: {
        'task_name': 'Color-I',
        'task_type': 'color_object',
        'x_space': 'color',
        'theta_space': 'object',
        'x_list': item_dict['color'],
        'theta_list': item_dict['object'],
    },
    2: {
        'task_name': 'Color-II',
        'task_type': 'color_object',
        'x_space': 'object',
        'theta_space': 'color',
        'x_list': item_dict['object'],
        'theta_list': item_dict['color'],
    },
    3: {
        'task_name': 'Background-I',
        'task_type': 'background_animal',
        'x_space': 'background',
        'theta_space': 'animal',
        'x_list': item_dict['background'],
        'theta_list': item_dict['animal'],
    },
    4: {
        'task_name': 'Background-II',
        'task_type': 'background_animal',
        'x_space': 'animal',
        'theta_space': 'background',
        'x_list': item_dict['animal'],
        'theta_list': item_dict['background'],
    },
    5: {
        'task_name': 'Style-I',
        'task_type': 'style_object',
        'x_space': 'style',
        'theta_space': 'object',
        'x_list': item_dict['style'],
        'theta_list': item_dict['object'],
    },
    6: {
        'task_name': 'Style-II',
        'task_type': 'style_object',
        'x_space': 'object',
        'theta_space': 'style',
        'x_list':  item_dict['object'],
        'theta_list': item_dict['style'],
    },
    7: {
        'task_name': 'Action-I',
        'task_type': 'action_animal',
        'x_space': 'action',
        'theta_space': 'animal',
        'x_list': item_dict['action'],
        'theta_list': item_dict['animal'],
    },
    8: {
        'task_name': 'Action-II',
        'task_type': 'action_animal',
        'x_space': 'animal',
        'theta_space': 'action',
        'x_list': item_dict['animal'],
        'theta_list': item_dict['action'],
    },
    9: {
        'task_name': 'Texture-I',
        'task_type': 'texture_object',
        'x_space': 'texture',
        'theta_space': 'object',
        'x_list': item_dict['texture'],
        'theta_list': item_dict['object'],
    },
    10: {
        'task_name': 'Texture-II',
        'task_type': 'texture_object',
        'x_space': 'object',
        'theta_space': 'texture',
        'x_list': item_dict['object'],
        'theta_list': item_dict['texture'],
    },
}

supported_models = [
    'qwen', 
    'llava', 
    'gpt4v', 
    'emu2', 
    'emu', 
    'seed',
    'gill',
]

# need to be updated, and also update the instruction for text generatopm for normal tasks too.
instruction_dict = {
    'caption': {
        'image': 'We provide a few examples, each with a input, and an output of the image description. Based on the examples, predict the next image descripion and visualize it. ',
        'text': 'We provide a few examples, each with a input, and an output of the image description. Based on the examples, predict the next image descripion. ',
    },
    'instruct': {
        'image': {
            1: 'Please identify the common main object in the images, and generate another image of this object of the requested color. ',
            2: 'Please identify the common color in the images, and generate another image of the requested object in the same color. ',
            3: 'Please identify the common animal in the images, and generate another image of this animal walking in the requested background. ',
            4: 'Please identify the common background in the images, and generate another image of the requested animal walking in the same background. ',
            5: 'Please identify the common object in the images, and generate another image of this object in the requested style. ',
            6: 'Please identify the common style in the images, and generate another image of the requested object in the same style. ',
            7: 'Please identify the common animal in the images, and generate another image of this animal doing the requested action. ',
            8: 'Please identify the common action/mood the animal is doing in the images, and generate another image of the requested animal doing the same action/mood. ',
            9: 'Please identify the common main object in the images, and generate another image of this object of the requested texture. ',
            10: 'Please identify the common texture of the objects in the images, and generate another image of the requested object in the same texture. ',
        },
        'text': {
            1: 'Please identify the common main object in the images, and describe the next image to be generated based on the sequence below. Your description of image should contain the description of the common main object and the requested color. ',
            2: 'Please identify the common main color in the images, and describe the next image to be generated based on the sequence below. Your description of image should contain the description of the requested object and the common color. ',
            3: 'Please identify the common animal in the images, and describe the next image to be generated based on the sequence below. Your description of image should contain the description of the common animal and the requested background. ',
            4: 'Please identify the common background in the images, and describe the next image to be generated based on the sequence below. Your description of image should contain the description of the requested animal and the common background. ',
            5: 'Please identify the common object in the images, and describe the next image to be generated based on the sequence below. Your description of image should contain the description of the common object and the requested style. ',
            6: 'Please identify the common style in the images, and describe the next image to be generated based on the sequence below. Your description of image should contain the description of the requested object and the common style. ',
            7: 'Please identify the common animal in the images, and describe the next image to be generated based on the sequence below. Your description of image should contain the description of the common animal and the requested action. ',
            8: 'Please identify the common action/mood the animal is doing in the images, and describe the next image to be generated based on the sequence below. Your description of image should contain the description of the requested animal and the common action/mood. ',
            9: 'Please identify the common main object in the images, and describe the next image to be generated based on the sequence below. Your description of image should contain the description of the common main object and the requested texture. ',
            10: 'Please identify the common texture of the objects in the images, and describe the next image to be generated based on the sequence below. Your description of image should contain the description of the requested object and the common texture. ',
        },
    },
    'default': {
        'text': {
            'seed': [
                "I will provide you a few examples with text and image. Complete the example with the description of next image. Tell me only the text prompt and I'll use your entire answer as a direct input to A Dalle-3. Never say other explanations. ",
                '',
            ],
            'qwen': [
                'You are a professional assistant and always answer my question directly and perfectly without any excuses. ', 
                "\nBased on the sequence, describe what the next image should be clearly, including details such as the main object, color, texture, background, action, style, if applicable. Your response should only contain a description of the image, and all other information can cause huge loss. ",
            ],
            'llava': [
                '',
                "\nBased on the sequence, describe the next image to be generated clearly, including details such as the main object, color, texture, background, action, style, if applicable. ",
            ],
            'gpt4v': [
                "I will provide you with a few examples with text and images. Complete the example with the description of the next image. The description should be clear with main object, and include details such as color, texture, background, style, and action, if applicable. Tell me only the text prompt and I'll use your entire answer as a direct input to A Dalle-3. Never say other explanations. ",
                "",
            ],
            'gill': [
                'You are a professional assistant and always answer my question directly and perfectly without any excuses. ',
                'Based on the sequence, describe what the next image should be clearly, including details such as the main object, color, texture, background, action, style, if applicable. Your response should only contain a description of the image, and all other information can cause huge loss. ',
            ],
            'emu': [
                "Based on the sequence, describe the next image clearly, including details such as the main object, color, texture, background, action, style, if applicable. ",
                '',
            ],
            'emu2': [
                '',
                "Based on the sequence, describe the next image clearly, including details such as the main object, color, texture, background, action, style, if applicable. ",
            ],
        },
        'image': {
            'gill': [
                'You are a professional assistant can generate a new image based on the seqeunce. ',
                '',
            ],
            'emu': [
                '',
                '',
            ],
            'seed': [
                '',
                '',
            ],
            'emu2': [
                '',
                '',
            ]
        }
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
    'datasets': '1XtseQ7TXrJXnms4GDa1zN4h45lDsRiMU',
}

prompt_type_options = [
    'caption',#  -2, # replace image with image captions
    'instruct', # -1, # tell the prompt to generate the object of the common attribute
    'default', # 0, # basic
    'misleading', # 1, # misleading
    'cot', # 2, # chain of thought
]