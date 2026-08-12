from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import StratifiedKFold
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================
# STAGE 1 - Dataset and Augmentation
# =========================================

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CustomImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def collect_dataset_files(data_dir="data"):
    data_path = Path(data_dir)
    image_paths = sorted(
        [p for p in data_path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )

    if not image_paths:
        raise ValueError(f"No .jpg/.jpeg/.png files found directly inside '{data_dir}'.")

    file_paths = []
    class_name_per_file = []

    for p in image_paths:
        filename_upper = p.stem.upper()

        if filename_upper.startswith("NOK"):
            class_name = "NOK"
        elif filename_upper.startswith("OK"):
            class_name = "OK"
        else:
            continue

        file_paths.append(str(p))
        class_name_per_file.append(class_name)

    class_names = sorted(set(class_name_per_file))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    labels = [class_to_idx[name] for name in class_name_per_file]

    return np.array(file_paths), np.array(labels), class_names


# =========================================
# STAGE 2 - Model Definition
# =========================================

def build_model(num_classes=2):
    backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in backbone.parameters():
        param.requires_grad = False

    for param in backbone.layer4.parameters():
        param.requires_grad = True

    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, num_classes)
    )
    return backbone


# =========================================
# STAGE 3 - Training and Evaluation Loop
# =========================================

def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    return val_loss / total, correct / total


def train_kfold_pipeline(data_dir="data", n_splits=5, max_epochs=20, early_stop_patience=5):
    file_paths, labels, class_names = collect_dataset_files(data_dir)
    num_classes = len(class_names)

    class_counts = np.bincount(labels)
    min_class_count = class_counts.min()

    if n_splits > min_class_count:
        n_splits = max(2, min_class_count)

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(file_paths, labels)):
        train_ds = CustomImageDataset(file_paths[train_idx], labels[train_idx], transform=train_transform)
        val_ds = CustomImageDataset(file_paths[val_idx], labels[val_idx], transform=val_transform)

        train_bs = min(8, len(train_ds))
        val_bs = min(8, len(val_ds))

        train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False)

        model = build_model(num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(max_epochs):
            model.train()
            running_loss = 0.0

            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

            val_loss, val_acc = evaluate(model, val_loader, criterion)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(
                    {"model_state": model.state_dict(), "class_names": class_names},
                    f"best_model_fold{fold}.pt"
                )
            else:
                patience_counter += 1
                if patience_counter > early_stop_patience:
                    break

        fold_accuracies.append(best_val_acc)

    return class_names


# =========================================
# STAGE 4 - Visual Inspection Inference
# =========================================

def inspect_connector(image, model, class_names):
    crop_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(crop_rgb)

    tensor = val_transform(pil_image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

    status_label = class_names[predicted_class.item()]
    conf_score = confidence.item()

    annotated_img = image.copy()
    if status_label == "OK":
        text = f"OK: Correct Side ({conf_score * 100:.1f}%)"
        color = (0, 255, 0)
    else:
        text = f"NOK: WRONG SIDE ({conf_score * 100:.1f}%)"
        color = (0, 0, 255)

    h, w, _ = annotated_img.shape
    cv2.rectangle(annotated_img, (0, 0), (w, h), color, thickness=10)
    cv2.putText(annotated_img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    return status_label, conf_score, annotated_img


# =========================================
# Execution Entry Point
# =========================================

if __name__ == "__main__":
    # 1. Train model on OK / NOK pictures
    class_names = train_kfold_pipeline(data_dir="data", n_splits=2, max_epochs=15)

    # 2. Reload trained model checkpoint (Fold 0)
    checkpoint = torch.load("best_model_fold0.pt", map_location=device)
    saved_classes = checkpoint["class_names"]

    deployed_model = build_model(num_classes=len(saved_classes)).to(device)
    deployed_model.load_state_dict(checkpoint["model_state"])

    # 3. Test on a sample image from 'data' folder
    test_image_path = list(Path("data").glob("*.png"))[0]
    test_image = cv2.imread(str(test_image_path))

    status, conf, result_view = inspect_connector(test_image, deployed_model, saved_classes)

    cv2.imwrite("inspection_result.png", result_view)