import pickle
import random
import torch
from DataPreprocessing import get_dataloaders
from TrainingModules import evaluate
from Utill import plot_sensitivity_scan, sensitivity_scan
from VGG import VGG
from Viewer import plot_weight_distribution  # Ensure you import your correct model architecture
seed=0
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = VGG()  # Replace with your actual model class
path="../mrleyedataset"
#model_path='./checkpoint/vgg_mrl_99.51375579833984.pth'
model_path='vgg_mrl_99.51.pth'
# Load the saved state_dict correctly
state_dict = torch.load(model_path, map_location=torch.device(device))  # Use 'cpu' if necessary
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
plot_weight_distribution(model)
train_dataloader,test_dataloader=get_dataloaders(path )
dense_model_accuracy=evaluate(model,test_dataloader)

sparsities, accuracies = sensitivity_scan(
    model, test_dataloader, scan_step=0.1, scan_start=0.1, scan_end=1.0)

with open("sparsities.pkl", "wb") as f:
    pickle.dump(sparsities, f)

with open("accuracies.pkl", "wb") as f:
    pickle.dump(accuracies, f)

plot_sensitivity_scan(sparsities, accuracies, dense_model_accuracy)