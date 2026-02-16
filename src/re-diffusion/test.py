import torch
import matplotlib.pyplot as plt
from diffusers import AutoencoderKL
from datasets import load_dataset
import torchvision.transforms as T
from PIL import Image
import requests
from io import BytesIO

dataset = load_dataset(
    "laion/aesthetics_v2_4.75",
    split="train",
    streaming=True  # IMPORTANT: dataset is huge
)

def filtered(dataset, min_score=5.0):
    for s in dataset:
        if s["AESTHETIC_SCORE"] >= min_score:
            yield s

good_dataset = filtered(dataset, min_score=5.0)

transform = T.Compose([
    T.Resize(512),
    T.CenterCrop(512),
    T.ToTensor(),  # [0,1]
])

device = "cpu"

# Load VAE
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse"
).to(device)
vae.eval()

# ---- SAFE SAMPLE FETCH ----
img = None
while img is None:
    sample = next(iter(good_dataset))
    try:
        print(sample.keys())
        url = sample["URL"]
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Skipping bad sample: {e}")
        img = None

print("Loaded image successfully")

# Preprocess
x = transform(img).unsqueeze(0).to(device)  # (1,3,H,W)
x = x * 2 - 1  # [0,1] → [-1,1]

# Encode
with torch.no_grad():
    posterior = vae.encode(x)
    z = posterior.latent_dist.mean
    z = z * vae.config.scaling_factor


print(z.shape)
# Decode
with torch.no_grad():
    recon = vae.decode(z / vae.config.scaling_factor).sample

recon = (recon.clamp(-1, 1) + 1) / 2

# Visualize
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Original (LAION URL)")
plt.imshow(x[0].permute(1,2,0).cpu() * 0.5 + 0.5)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("VAE Reconstruction")
plt.imshow(recon[0].permute(1,2,0).cpu())
plt.axis("off")
plt.show()
