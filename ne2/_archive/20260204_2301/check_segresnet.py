import torch
import sys
sys.path.append("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/all_four_validations/mandatory_artifacts_segresnet")

# Load the checkpoint to see its structure
checkpoint = torch.load("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/all_four_validations/mandatory_artifacts_segresnet/best_resumed.pt", weights_only=False)

print("Checkpoint keys:", list(checkpoint.keys()))
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print("Model state dict keys (first 10):", list(state_dict.keys())[:10])
else:
    state_dict = checkpoint
    print("State dict keys (first 10):", list(state_dict.keys())[:10])

print("Total parameters:", len(state_dict.keys()))
