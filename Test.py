import random
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch
from DataPreprocessing import get_dataloaders,train_transform,test_transform
from Model_Evaluation import get_model_macs, get_model_size, get_model_sparsity
from Utill import get_labels_preds, print_model
from VGG import VGG
from TrainingModules import evaluate  # Ensure you import your correct model architecture
import torch

#fix the randomness
seed=0
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model and tensors to device

path='./mrleyedataset'
# Initialize the model
model = VGG()  # Replace with your actual model class

#model_path='./checkpoint/vgg_mrl_99.51375579833984.pth'
model_path='vgg_mrl_99.51.pth'
# Load the saved state_dict correctly
state_dict = torch.load(model_path, map_location=torch.device(device))  # Use 'cpu' if necessary
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

# Print out missing/unexpected keys for debugging
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)

train_dataloader,test_dataloader=get_dataloaders(path, train_transform=train_transform, test_transform=test_transform)
########################################test model#####################################
# Set model to evaluation mode
# model.eval()
input_tensor=torch.randn(1, 3, 80, 80)
# output = model(input_tensor)  # Ensure input_tensor is properly formatted
#######################################################################################

########################################test model#####################################
metric,_ = evaluate(model, test_dataloader)
print("accuracy:",metric)
#######################################################################################
######################################## model metrics ################################
print_model(model)
macs =get_model_macs(model,input_tensor)
sparsity =get_model_sparsity(model)
model_size =get_model_size(model)
print('macs:',macs)
print('sparsity:',sparsity)
print('model size:',model_size)
#######################################################################################
all_labels, all_preds,all_outputs,loss = get_labels_preds(model,test_dataloader)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
conf_matrix=confusion_matrix(all_labels, all_preds)
print(conf_matrix)