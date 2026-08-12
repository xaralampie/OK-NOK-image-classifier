import json
import os
import torch
import torch.nn as nn
from torchvision import models

# List of PyTorch checkpoint files to export
CHECKPOINTS = [
    "black_connector.pt",
    "white_connector.pt",
    "green_connector.pt",
]


def build_model(num_classes):
  """Reconstructs the ResNet18 model architecture matching train_custom.py."""
  backbone = models.resnet18(weights=None)
  in_features = backbone.fc.in_features
  backbone.fc = nn.Sequential(
      nn.Dropout(0.2),
      nn.Linear(in_features, 128),
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.Linear(128, num_classes),
  )
  return backbone


def convert_pt_to_onnx(pt_path):
  """Converts a single .pt file to .onnx and exports its classes to JSON."""
  if not os.path.exists(pt_path):
    print(f"[SKIP] File not found: {pt_path}")
    return

  # Create output filenames based on input filename
  base_name = os.path.splitext(pt_path)[0]  # e.g. "black_connector"
  onnx_path = f"{base_name}.onnx"  # e.g. "black_connector.onnx"
  json_path = f"{base_name}_classes.json"  # e.g. "black_connector_classes.json"

  print(f"\n--- Processing: {pt_path} ---")

  # 1. Load PyTorch Checkpoint
  checkpoint = torch.load(pt_path, map_location="cpu")
  class_names = checkpoint["class_names"]
  print(f"  Classes ({len(class_names)}): {class_names}")

  # 2. Build Model & Load Weights
  model = build_model(num_classes=len(class_names))
  model.load_state_dict(checkpoint["model_state"])
  model.eval()

  # 3. Create dummy input tensor (batch=1, channels=3, height=224, width=224)
  dummy_input = torch.randn(1, 3, 224, 224)

  # 4. Export ONNX
  torch.onnx.export(
      model,
      dummy_input,
      onnx_path,
      input_names=["input"],
      output_names=["output"],
      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
  )
  print(f"  [SUCCESS] Exported ONNX: {onnx_path}")

  # 5. Export Class Names to JSON
  with open(json_path, "w") as f:
    json.dump(class_names, f, indent=2)
  print(f"  [SUCCESS] Saved Classes: {json_path}")


def main():
  print("=" * 50)
  print(" Starting Batch PyTorch to ONNX Conversion")
  print("=" * 50)

  for pt_file in CHECKPOINTS:
    convert_pt_to_onnx(pt_file)

  print("\n" + "=" * 50)
  print(" Batch Export Complete!")
  print("=" * 50)
  print("\nCopy the following 6 files to your Raspberry Pi:")
  for pt_file in CHECKPOINTS:
    base = os.path.splitext(pt_file)[0]
    if os.path.exists(f"{base}.onnx"):
      print(f" - {base}.onnx")
      print(f" - {base}_classes.json")


if __name__ == "__main__":
  main()