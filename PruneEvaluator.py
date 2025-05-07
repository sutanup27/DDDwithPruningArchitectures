

import copy
from matplotlib import path
import torch
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *

from DataPreprocessing import get_dataloaders
from TrainingModules import evaluate
from VGG import VGG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path='../mrleyedataset'
model=VGG()
model_path='vgg_mrl_99.09.pth'
# Load the saved state_dict correctly
state_dict = torch.load(model_path, map_location=torch.device(device),weights_only=False)  # Use 'cpu' if necessary
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
model.to(device)
train_dataloader,test_dataloader=get_dataloaders(path, batch_size=64 ) # Basemodel
dense_model_accuracy=evaluate(model,test_dataloader)
print('dense_model_accuracy:',dense_model_accuracy)
pruned_model=copy.deepcopy(model)

torch.save(pruned_model,'model.path')
loaded_model=torch.load('model.path',map_location=torch.device(device))

loaded_model_accuracy=evaluate(loaded_model,test_dataloader)
print('loaded_model_accuracy:',loaded_model_accuracy)
