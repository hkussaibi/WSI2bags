import os
import torch
import openslide
from torchvision import transforms
from transformers import AutoImageProcessor, ViTModel
from tqdm import tqdm
from utils import get_valid_patches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
phikon = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False).to(device)
data_transform = transforms.Compose([transforms.ToTensor()])

data_dir = "./data"
output_dir = "./data/features"
os.makedirs(output_dir, exist_ok=True)

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    for wsi_file in os.listdir(class_path):
        if not wsi_file.endswith(".svs"):
            continue
        slide_path = os.path.join(class_path, wsi_file)
        slide = openslide.open_slide(slide_path)
        thumbnail = slide.get_thumbnail((500, 500))
        patches = get_valid_patches(slide, thumbnail, tissue_threshold=0.9)
        tensors = [data_transform(patch) for patch in patches]
        loader = torch.utils.data.DataLoader(tensors, batch_size=64)
        features = []
        for batch in tqdm(loader, desc=f"Extracting {wsi_file}"):
            inputs = image_processor(batch, return_tensors="pt", do_rescale=False).to(device)
            with torch.no_grad():
                outputs = phikon(**inputs)
                batch_features = outputs.last_hidden_state[:, 0, :]
                features.append(batch_features.cpu())
        features = torch.cat(features)
        torch.save(features, os.path.join(output_dir, f"{class_name}_{wsi_file[:-4]}.pth"))