import torch
# You need to import/recreate your model architecture
# from your_model_definition import YourModelClass  # Replace with your actual import
from models.encoder import SignatureEncoder

# Load trained model
model = SignatureEncoder()
# Use same args as when you saved it

# Load the state dict
checkpoint = torch.load('final_model.pt', map_location='cpu')

# Handle different checkpoint formats
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    # Assume the checkpoint itself is the state dict
    state_dict = checkpoint

model.load_state_dict(state_dict)

# Count parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
total_params = trainable_params + non_trainable_params

print(f"Trainable parameters: {trainable_params:,}")
print(f"Non-trainable parameters: {non_trainable_params:,}")
print(f"Total parameters: {total_params:,}")