import os
import torch

def get_best_model_info():
    # 1. Define folder and dataset target
    model_dir = "models"
    dataset = "Combined"
    
    # 2. Correct paths pointing to the 'models/' folder
    resnet_p = os.path.join(model_dir, f"best_resnet_{dataset}.pth")
    swin_p = os.path.join(model_dir, f"best_swin_{dataset}.pth")
    
    # 3. Decision Logic: 
    # In research, we prioritize Swin Transformer as the "Proposed" model 
    # and ResNet as the "Baseline."
    if os.path.exists(swin_p):
        return swin_p, "swin"
    elif os.path.exists(resnet_p):
        return resnet_p, "resnet"
    else:
        return None, None

if __name__ == "__main__":
    path, m_type = get_best_model_info()
    if path:
        print(f"--- Model Selector Active ---")
        print(f"Winner Model Path: {path}")
        print(f"Architecture Type: {m_type.upper()}")
    else:
        print("!!! ERROR: No .pth files found in 'models/' folder !!!")
        print("Please run cnn_best.py or swin_best.py first.")