import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torchvision import models
import timm
import matplotlib.pyplot as plt
from PIL import Image

def swin_reshape_transform(tensor):
    if len(tensor.shape) == 4:
        return tensor.transpose(2, 3).transpose(1, 2)
    batch_size, num_tokens, channels = tensor.shape
    height = width = int(np.sqrt(num_tokens))
    result = tensor.reshape(batch_size, height, width, channels)
    return result.transpose(2, 3).transpose(1, 2)

def run_gradcam(model_type="resnet", img_path="test.jpg"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs", exist_ok=True)

    # 1. Setup Classes
    class_names = sorted(os.listdir("data/dataset_combined"))
    num_classes = len(class_names)
    
    # Load original image
    full_img_pil = Image.open(img_path).convert('RGB')
    full_img_np = np.array(full_img_pil).astype(np.float32) / 255.0
    orig_w, orig_h = full_img_pil.size

    # 2. CREATE LEAF MASK (Removes background noise)
    gray = cv2.cvtColor((full_img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 242, 255, cv2.THRESH_BINARY_INV)
    mask_float = cv2.GaussianBlur(mask.astype(float) / 255.0, (15, 15), 0)

    # 3. Load Model
    if model_type == "resnet":
        model = models.resnet50()
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes))
        model.load_state_dict(torch.load("models/best_resnet_Combined.pth", map_location=device))
        target_layers = [model.layer4[-1]]
        method = GradCAMPlusPlus 
        reshape_transform = None
    else:
        model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=num_classes)
        model.load_state_dict(torch.load("models/best_swin_Combined.pth", map_location=device))
        # Targeted layer for sharper Swin visuals
        target_layers = [model.layers[-1].blocks[-1].norm1]
        method = GradCAM
        reshape_transform = swin_reshape_transform

    model = model.to(device).eval()

    # 4. Predict
    img_224 = cv2.resize(full_img_np, (224, 224))
    input_tensor = preprocess_image(img_224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        pred_idx = output.argmax(1).item()
        pred_name = class_names[pred_idx]
        conf = probabilities[pred_idx].item()

    # 5. Generate Heatmap
    cam_engine = method(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    grayscale_cam = cam_engine(input_tensor=input_tensor, targets=None)[0, :]

    # 6. RESEARCH REFINEMENT: SIGMOID SHARPENING & MASKING
    # Upscale to full resolution
    heatmap_full = cv2.resize(grayscale_cam, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    
    # Contrast Enhancement: Sharpens the blobs on lesions
    heatmap_full = 1 / (1 + np.exp(-15 * (heatmap_full - 0.5))) 
    
    # Smooth the result
    heatmap_full = cv2.GaussianBlur(heatmap_full, (31, 31), 0)

    # Apply Mask (Force background to zero/blue)
    heatmap_full = heatmap_full * mask_float

    # 7. Final Overlay
    visualization = show_cam_on_image(full_img_np, heatmap_full, use_rgb=True)

    # 8. Create Final Figure
    plt.figure(figsize=(18, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(full_img_pil)
    plt.title(f"Original Input Leaf ({orig_w}x{orig_h})", fontsize=15)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f"Refined Grad-CAM ({model_type.upper()})\nPrediction: {pred_name} ({conf*100:.2f}%)", fontsize=15)
    plt.axis('off')

    output_fn = f"outputs/gradcam_refined_{model_type}.png"
    plt.tight_layout()
    plt.savefig(output_fn, dpi=300)
    plt.close()
    print(f"--- SUCCESS: Refined {model_type} saved to {output_fn} ---")

if __name__ == "__main__":
    run_gradcam("resnet")
    run_gradcam("swin")