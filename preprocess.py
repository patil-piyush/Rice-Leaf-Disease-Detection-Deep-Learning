import os, shutil, random

original_dir = r"data\Original Dataset"
aug_dir = r"data\Augmented Dataset"
sampled_dir = r"data\sampled_augmented"
combined_dir = r"data\dataset_combined"

os.makedirs(sampled_dir, exist_ok=True)
os.makedirs(combined_dir, exist_ok=True)

classes = os.listdir(original_dir)

# SAMPLE AUGMENTED (250 per class)
for cls in classes:
    src = os.path.join(aug_dir, cls)
    dst = os.path.join(sampled_dir, cls)
    os.makedirs(dst, exist_ok=True)

    imgs = os.listdir(src)
    sample = random.sample(imgs, min(250, len(imgs)))

    for img in sample:
        shutil.copy(os.path.join(src, img), os.path.join(dst, img))

# COMBINE DATASET
for cls in classes:
    comb_cls = os.path.join(combined_dir, cls)
    os.makedirs(comb_cls, exist_ok=True)

    # original
    for img in os.listdir(os.path.join(original_dir, cls)):
        shutil.copy(os.path.join(original_dir, cls, img), comb_cls)

    # sampled aug
    for img in os.listdir(os.path.join(sampled_dir, cls)):
        shutil.copy(os.path.join(sampled_dir, cls, img), comb_cls)

print("Preprocessing Done ")