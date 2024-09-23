# save_models.py

import torch
from torchvision.models import resnet50
from model_definitions3 import ModelA, ModelB

# Load the pretrained ResNet50 model
original_model = resnet50(pretrained=True)

# Instantiate Model A and Model B
model_a = ModelA(original_model)
model_b = ModelB(original_model)

# Save Model A
torch.save(model_a.state_dict(), 'model_a3.pth')

# Save Model B
torch.save(model_b.state_dict(), 'model_b3.pth')

print("Models have been saved successfully.")
