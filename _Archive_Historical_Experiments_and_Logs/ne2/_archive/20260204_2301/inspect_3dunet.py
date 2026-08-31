import torch
import sys

# Load 3D U-Net checkpoint
checkpoint_path = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\mandatory_artifacts_3dUnet\checkpoints\best_model.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("Checkpoint keys:", list(checkpoint.keys()))
print("\nEpoch:", checkpoint.get('epoch', 'N/A'))
print("Best val dice:", checkpoint.get('best_val_dice', 'N/A'))

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

print("\n=== State Dict Analysis ===")
print(f"Total keys: {len(state_dict)}")

# Group keys by layer type
keys_by_prefix = {}
for key in state_dict.keys():
    # Get first few parts of the key
    parts = key.split('.')
    prefix = '.'.join(parts[:3]) if len(parts) >= 3 else '.'.join(parts[:2])
    if prefix not in keys_by_prefix:
        keys_by_prefix[prefix] = []
    keys_by_prefix[prefix].append(key)

print("\n=== Layer Structure (first 20 prefixes) ===")
for i, (prefix, keys) in enumerate(sorted(keys_by_prefix.items())[:20]):
    print(f"{prefix}: {len(keys)} params")
    if i < 5:
        for k in keys[:3]:
            print(f"  - {k}: {state_dict[k].shape}")

# Look for clues about architecture
print("\n=== Looking for architecture clues ===")
sample_keys = list(state_dict.keys())[:10]
for k in sample_keys:
    print(f"{k}: {state_dict[k].shape}")
