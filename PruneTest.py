import copy
import random
import torch
from DataPreprocessing import get_dataloaders
from Model_Evaluation import get_model_size, get_sparsity
from TrainingModules import evaluate
from Utill import FineGrainedPruner, plot_sensitivity_scan, sensitivity_scan
from VGG import VGG
from Viewer import plot_weight_distribution  # Ensure you import your correct model architecture
seed=0
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

