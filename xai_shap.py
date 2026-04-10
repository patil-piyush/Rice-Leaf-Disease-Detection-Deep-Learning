import torch
import torch.nn as nn
import numpy as np
import shap
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image
import os
import cv2

def run_shap(img_path="test.jpg"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs", exist_ok=True)

    # 1. Load Classes and Model
    dataset_path = "data/dataset_combined"
    class_names = sorted(os.listdir(dataset_path))
    num_classes = len(class_names)
    
    model = models.resnet50()
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes))
    
    checkpoint = "models/best_resnet_Combined.pth"
    if not os.path.exists(checkpoint):
        print(f"Error: {checkpoint} not found.")
        return
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device).eval()

    # 2. Image Loading (Full Resolution vs Model Input)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    model_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return
    
    full_img_pil = Image.open(img_path).convert('RGB')
    orig_w, orig_h = full_img_pil.size
    input_tensor = model_transform(full_img_pil).unsqueeze(0).to(device)

    # 3. Predict Confidence (The "WHY" explanation)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, 3) # Get top 3 candidates

    pred_idx = top_idxs[0].item()
    pred_label = class_names[pred_idx]
    
    print(f"\n--- RESEARCH EXPLANATION FOR {img_path} ---")
    print(f"Primary Selection: {pred_label} ({top_probs[0]*100:.2f}% confidence)")
    print(f"Alternative 1: {class_names[top_idxs[1]]} ({top_probs[1]*100:.2f}%)")
    print(f"Alternative 2: {class_names[top_idxs[2]]} ({top_probs[2]*100:.2f}%)")
    print("-" * 45)

    # 4. Calculate SHAP (Spatial Evidence)
    background = torch.zeros_like(input_tensor).to(device)
    explainer = shap.GradientExplainer(model, background)
    print("Calculating SHAP spatial evidence Map...")
    shap_values = explainer.shap_values(input_tensor)

    # 5. Extract Heatmap correctly based on SHAP version
    if isinstance(shap_values, list):
        target_shap = shap_values[pred_idx]
    else:
        if shap_values.shape[1] == num_classes: target_shap = shap_values[:, pred_idx, :, :, :]
        elif shap_values.shape[-1] == num_classes: target_shap = shap_values[..., pred_idx]
        else: target_shap = shap_values

    target_shap = np.squeeze(target_shap)
    if target_shap.shape[-1] == 3: target_shap = np.transpose(target_shap, (2, 0, 1))
    heatmap_224 = np.sum(np.abs(target_shap), axis=0)

    # 6. Map Evidence to Full-Resolution Image
    heatmap_full = cv2.resize(heatmap_224, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    heatmap_full = cv2.GaussianBlur(heatmap_full, (51, 51), 0)
    heatmap_full = (heatmap_full - heatmap_full.min()) / (heatmap_full.max() - heatmap_full.min() + 1e-8)

    # 7. Create Visualization with full-size images
    plt.figure(figsize=(18, 10))
    
    plt.subplot(1, 2, 1)
    plt.imshow(full_img_pil)
    plt.title(f"Original High-Res Leaf\n({orig_w}x{orig_h})", fontsize=15, pad=15)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(full_img_pil)
    # Overlay the evidence map
    plt.imshow(heatmap_full, cmap='jet', alpha=0.45) 
    plt.title(f"SHAP Evidence Map for '{pred_label}'\nConfidence Score: {top_probs[0]*100:.2f}%", fontsize=15, pad=15)
    plt.axis('off')

    plt.tight_layout()
    output_fn = "outputs/shap_full_explanation.png"
    plt.savefig(output_fn, dpi=300)
    
    print(f"\nCONCLUSION FOR PAPER:")
    print(f"The image was selected as '{pred_label}' because the model detected specific patterns ")
    print(f"highlighted in RED on the SHAP Evidence Map. The model is {top_probs[0]*100:.2f}% certain.")
    print(f"--- SUCCESS: Results saved to {output_fn} ---")

if __name__ == "__main__":
    run_shap("test.jpg")