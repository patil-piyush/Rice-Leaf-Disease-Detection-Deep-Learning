import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Swin System Active: {device} ---")

    # FIXED: RandomErasing must come AFTER ToTensor()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(), 
        transforms.RandomErasing(p=0.2), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    summary_table = []
    os.makedirs("models", exist_ok=True)

    paths = {"Original": "data/Original Dataset", "Combined": "data/dataset_combined"}

    for name, path in paths.items():
        if not os.path.exists(path): continue
        print(f"\nTraining Swin on {name}...")
        
        ds = datasets.ImageFolder(path, transform=transform)
        tr_len = int(0.7 * len(ds)); vl_len = int(0.15 * len(ds)); ts_len = len(ds) - tr_len - vl_len
        train_ds, val_ds, test_ds = random_split(ds, [tr_len, vl_len, ts_len])
        
        loaders = {'train': DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2),
                   'val': DataLoader(val_ds, batch_size=16, shuffle=False),
                   'test': DataLoader(test_ds, batch_size=16, shuffle=False)}

        model = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=True, 
            num_classes=len(ds.classes),
            drop_rate=0.3,
            drop_path_rate=0.2
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        history = {'t_acc': [], 'v_acc': [], 't_loss': []}
        best_acc = 0

        for epoch in range(20):
            model.train()
            t_l, t_c, t_t = 0, 0, 0
            for x, y in loaders['train']:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x); loss = criterion(out, y)
                loss.backward(); optimizer.step()
                t_l += loss.item(); t_t += y.size(0); t_c += (out.argmax(1) == y).sum().item()

            model.eval()
            v_c, v_t = 0, 0
            with torch.no_grad():
                for x, y in loaders['val']:
                    x, y = x.to(device), y.to(device)
                    v_c += (model(x).argmax(1) == y).sum().item(); v_t += y.size(0)
            
            history['t_acc'].append(t_c/t_t); history['v_acc'].append(v_c/v_t); history['t_loss'].append(t_l/len(loaders['train']))
            print(f"Epoch {epoch+1:02d} | Tr-Acc: {history['t_acc'][-1]:.4f} | Val-Acc: {history['v_acc'][-1]:.4f}")

            if history['v_acc'][-1] > best_acc:
                best_acc = history['v_acc'][-1]
                torch.save(model.state_dict(), f"models/best_swin_{name}.pth")

        # FINAL TEST
        model.load_state_dict(torch.load(f"models/best_swin_{name}.pth"))
        model.eval()
        ts_c, ts_t = 0, 0
        with torch.no_grad():
            for x, y in loaders['test']:
                x, y = x.to(device), y.to(device)
                ts_c += (model(x).argmax(1) == y).sum().item(); ts_t += y.size(0)
        
        summary_table.append([name, history['t_acc'][-1], history['t_loss'][-1], ts_c/ts_t])

    print("\n" + "="*75)
    print(f"{'Swin Dataset':<15} | {'Final Tr. Acc':<15} | {'Tr. Loss':<12} | {'Test Acc':<12}")
    print("-" * 75)
    for row in summary_table:
        print(f"{row[0]:<15} | {row[1]:<15.4f} | {row[2]:<12.4f} | {row[3]:<12.4f}")
    print("="*75)

if __name__ == "__main__":
    main()