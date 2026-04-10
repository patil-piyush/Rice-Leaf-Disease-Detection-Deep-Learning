import os
import shutil

def main():
    target = "data/dataset_gan_hybrid"
    os.makedirs(target, exist_ok=True)
    # Merges Original (Real) + GAN (Synthetic)
    for src in ["data/Original Dataset", "data/gan_generated"]:
        for cls in os.listdir(src):
            os.makedirs(os.path.join(target, cls), exist_ok=True)
            for img in os.listdir(os.path.join(src, cls)):
                shutil.copy(os.path.join(src, cls, img), os.path.join(target, cls, img))
    print(f"--- GAN Hybrid Dataset Created at {target} ---")

if __name__ == "__main__":
    main()