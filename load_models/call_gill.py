import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'models/gill'))

from gill import models
from gill import utils

def load_gill(device = 'cuda'):
    # Download the model checkpoint and embeddings to checkpoints/gill_opt/
    model_dir = f'{root_dir}/models/gill/checkpoints/gill_opt/'
    model = models.load_gill(model_dir, device = device)
    return model

def call_