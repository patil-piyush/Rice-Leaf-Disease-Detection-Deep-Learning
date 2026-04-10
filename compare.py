import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Research Comparison Engine: {device} ---")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_path = "data/dataset_combined"
    if not os.path.exists(data_path):
        print("Error: Dataset not found.")
        return

    full_ds = datasets.ImageFolder(data_path, transform=transform)
    num_classes = len(full_ds.classes)
    
    tr_len = int(0.7 * len(full_ds))
    vl_len = int(0.15 * len(full_ds))
    ts_len = len(full_ds) - tr_len - vl_len
    _, _, test_ds = random_split(full_ds, [tr_len, vl_len, ts_len])
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    results = []

    # 1. EVALUATE RESNET50 (Updated to match Anti-Overfit Architecture)
    resnet_path = "models/best_resnet_Combined.pth"
    if os.path.exists(resnet_path):
        print("Evaluating ResNet50...")
        model = models.resnet50()
        
        # MUST MATCH TRAINING: Dropout + Linear
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(num_ftrs, num_classes)
        )
        
        model.load_state_dict(torch.load(resnet_path, map_location=device))
        model = model.to(device).eval()

        preds, labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                preds.extend(model(x).argmax(1).cpu().numpy())
                labels.extend(y.numpy())
        results.append(["ResNet50", accuracy_score(labels, preds), f1_score(labels, preds, average='macro')])

    # 2. EVALUATE SWIN TINY
    swin_path = "models/best_swin_Combined.pth"
    if os.path.exists(swin_path):
        print("Evaluating Swin Tiny...")
        model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=num_classes)
        model.load_state_dict(torch.load(swin_path, map_location=device))
        model = model.to(device).eval()

        preds, labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                preds.extend(model(x).argmax(1).cpu().numpy())
                labels.extend(y.numpy())
        results.append(["Swin Tiny", accuracy_score(labels, preds), f1_score(labels, preds, average='macro')])

    print("\n" + "="*70)
    print(f"{'Model Architecture':<25} | {'Test Accuracy':<15} | {'F1-Score':<15}")
    print("-" * 70)
    for res in results:
        print(f"{res[0]:<25} | {res[1]:<15.4f} | {res[2]:<15.4f}")
    print("="*70)

if __name__ == "__main__":
    main()