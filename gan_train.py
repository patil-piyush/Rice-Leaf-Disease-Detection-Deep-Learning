import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
import os

# --- Hyperparameters ---
L_DIM = 100
IMG_SIZE = 64 
BATCH_SIZE = 64
EPOCHS = 100 # Increased for better texture detail
N_CLASSES = 8

# 1. GENERATOR: Merges class info with noise at the start
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(N_CLASSES, N_CLASSES)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(L_DIM + N_CLASSES, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate noise [Batch, 100, 1, 1] and labels [Batch, 8, 1, 1]
        cls_info = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        gen_input = torch.cat((noise, cls_info), 1)
        return self.model(gen_input)

# 2. DISCRIMINATOR: Uses InstanceNorm for better texture learning
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(N_CLASSES, N_CLASSES)
        self.model = nn.Sequential(
            nn.Conv2d(3 + N_CLASSES, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False)
        )

    def forward(self, img, labels):
        # Create a label map that matches the image HxW
        lvl = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        lvl = lvl.expand(-1, -1, img.size(2), img.size(3)) # Dynamically matches 64x64
        d_in = torch.cat((img, lvl), 1)
        return self.model(d_in).view(-1)

# 3. Gradient Penalty to prevent blurriness (WGAN-GP)
def compute_gradient_penalty(D, real_samples, fake_samples, labels, device):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = torch.ones(d_interpolates.shape, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Improved GAN Engine Active: {device} ---")
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder("data/Original Dataset", transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    class_names = dataset.classes

    netG, netD = Generator().to(device), Discriminator().to(device)
    optG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.9))
    optD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.9))

    for epoch in range(EPOCHS):
        for i, (real_imgs, labels) in enumerate(loader):
            b_size = real_imgs.size(0)
            real_imgs, labels = real_imgs.to(device), labels.to(device)

            # --- Train Discriminator ---
            optD.zero_grad()
            noise = torch.randn(b_size, L_DIM, 1, 1, device=device)
            fake_imgs = netG(noise, labels)
            
            d_loss = -torch.mean(netD(real_imgs, labels)) + torch.mean(netD(fake_imgs.detach(), labels))
            gp = compute_gradient_penalty(netD, real_imgs, fake_imgs.detach(), labels, device)
            (d_loss + gp).backward()
            optD.step()

            # --- Train Generator ---
            if i % 5 == 0:
                optG.zero_grad()
                g_loss = -torch.mean(netD(fake_imgs, labels))
                g_loss.backward()
                optG.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | D_Loss: {d_loss.item():.4f}")

    # Generate 250 images per class
    out_path = "data/gan_generated"
    print("\n--- Generating Clean Synthetic Dataset ---")
    netG.eval()
    with torch.no_grad():
        for idx, name in enumerate(class_names):
            os.makedirs(f"{out_path}/{name}", exist_ok=True)
            for j in range(250):
                noise = torch.randn(1, L_DIM, 1, 1, device=device)
                lab = torch.tensor([idx], device=device)
                sample = netG(noise, lab)
                utils.save_image(sample, f"{out_path}/{name}/gan_{j}.png", normalize=True)
    print("DONE: Synthetic data created.")

if __name__ == "__main__":
    main()
    