import sys
import torch
import requests
from pathlib import Path

# URL for TorchVision ResNet50 (IMAGENET1K_V2 - DEFAULT in torchvision 0.15.2)
URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"

def download_resnet50_weights():
    hub_dir = Path(torch.hub.get_dir()) / "checkpoints"
    hub_dir.mkdir(parents=True, exist_ok=True)
    target_path = hub_dir / "resnet50-11ad3fa6.pth"
    
    if target_path.exists() and target_path.stat().st_size > 90 * 1024 * 1024:
        print(f"Weights already present at: {target_path} ({target_path.stat().st_size / (1024**2):.2f} MB)")
        return
        
    print(f"Downloading ResNet50 weights to: {target_path} ...")
    headers = {"User-Agent": "torchvision/0.15.2"}
    response = requests.get(URL, headers=headers, stream=True, timeout=120)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 2 * 1024 * 1024  # 2 MB chunks
    
    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                print(f"  Downloaded: {downloaded / (1024**2):.2f} MB / {total_size / (1024**2):.2f} MB ({percent:.1f}%)")
                
    print(f"\n[SUCCESS] Download completed! Saved to {target_path}")

if __name__ == "__main__":
    download_resnet50_weights()
