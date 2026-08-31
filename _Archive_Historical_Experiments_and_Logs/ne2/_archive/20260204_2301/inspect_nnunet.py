import torch

# Load nnU-Net checkpoint
checkpoint_path = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\mandatory_artifacts_nnunet\checkpoints\fold_0\checkpoint_best.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("=== nnU-Net Checkpoint ===")
print("Keys:", list(checkpoint.keys())[:10])

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print("Found 'model_state_dict' key")
else:
    state_dict = checkpoint
    print("Using checkpoint directly as state_dict")

print(f"\nTotal keys: {len(state_dict)}")

# Show first 20 keys
print("\n=== First 20 Keys ===")
for i, key in enumerate(list(state_dict.keys())[:20]):
    print(f"{key}: {state_dict[key].shape}")

# Analyze structure
print("\n=== Layer Structure ===")
key_prefixes = {}
for key in state_dict.keys():
    parts = key.split('.')
    prefix = parts[0]
    if prefix not in key_prefixes:
        key_prefixes[prefix] = 0
    key_prefixes[prefix] += 1

for prefix, count in sorted(key_prefixes.items()):
    print(f"{prefix}: {count} params")
