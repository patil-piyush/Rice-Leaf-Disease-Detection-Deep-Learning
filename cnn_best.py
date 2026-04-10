import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- CNN System Active: Using {device} ---")
    
    # STRONGER AUGMENTATION TO PREVENT MEMORIZATION
    transform = transforms.Compose([
        transforms.Resize((230, 230)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    summary_table = []
    paths = {"Original": "data/Original Dataset", "Combined": "data/dataset_combined"}

    for name, path in paths.items():
        if not os.path.exists(path): continue
        print(f"\nTraining ResNet50 on {name}...")
        
        ds = datasets.ImageFolder(path, transform=transform)
        tr_len = int(0.7 * len(ds)); vl_len = int(0.15 * len(ds)); ts_len = len(ds) - tr_len - vl_len
        train_ds, val_ds, test_ds = random_split(ds, [tr_len, vl_len, ts_len])

        loaders = {
            'train': DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2),
            'val': DataLoader(val_ds, batch_size=32, shuffle=False),
            'test': DataLoader(test_ds, batch_size=32, shuffle=False)
        }

        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # ADDING DROPOUT LAYER TO PREVENT OVERFITTING
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(num_ftrs, len(ds.classes))
        )
        model = model.to(device)

        # INCREASED WEIGHT DECAY (from 0.01 to 0.05)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        criterion = nn.CrossEntropyLoss()
        
        history = {'t_loss': [], 'v_loss': [], 't_acc': [], 'v_acc': []}
        best_acc = 0

        for epoch in range(30):
            model.train()
            t_loss, t_corr, t_tot = 0, 0, 0
            for imgs, lbls in loaders['train']:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                out = model(imgs); loss = criterion(out, lbls)
                loss.backward(); optimizer.step()
                t_loss += loss.item(); t_tot += lbls.size(0); t_corr += (out.argmax(1) == lbls).sum().item()

            model.eval()
            v_loss, v_corr, v_tot = 0, 0, 0
            with torch.no_grad():
                for imgs, lbls in loaders['val']:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    out = model(imgs); v_loss += criterion(out, lbls).item()
                    v_tot += lbls.size(0); v_corr += (out.argmax(1) == lbls).sum().item()

            history['t_acc'].append(t_corr/t_tot); history['v_acc'].append(v_corr/v_tot)
            history['t_loss'].append(t_loss/len(loaders['train'])); history['v_loss'].append(v_loss/len(loaders['val']))

            if history['v_acc'][-1] > best_acc:
                best_acc = history['v_acc'][-1]
                torch.save(model.state_dict(), f"best_resnet_{name}.pth")

            print(f"Epoch {epoch+1:02d} | Tr-Acc: {history['t_acc'][-1]:.4f} | Val-Acc: {history['v_acc'][-1]:.4f}")

        # Final Test calculation for the table
        model.load_state_dict(torch.load(f"best_resnet_{name}.pth"))
        model.eval()
        ts_corr, ts_tot = 0, 0
        with torch.no_grad():
            for imgs, lbls in loaders['test']:
                imgs, lbls = imgs.to(device), lbls.to(device)
                ts_corr += (model(imgs).argmax(1) == lbls).sum().item(); ts_tot += lbls.size(0)
        
        summary_table.append([name, history['t_acc'][-1], history['t_loss'][-1], ts_corr/ts_tot])

    print("\n" + "="*70)
    print(f"{'Dataset':<15} | {'Final Tr. Acc':<15} | {'Tr. Loss':<12} | {'Test Acc':<12}")
    print("-" * 70)
    for row in summary_table:
        print(f"{row[0]:<15} | {row[1]:<15.4f} | {row[2]:<12.4f} | {row[3]:<12.4f}")
    print("="*70)

if __name__ == "__main__":
    main()