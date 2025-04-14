import os
import torch

features_dir = "./data/features"
bags_dir = "./data/bags"
os.makedirs(bags_dir, exist_ok=True)

chunk_size = 20
all_chunks, all_labels = [], []

for feature_file in os.listdir(features_dir):
    features = torch.load(os.path.join(features_dir, feature_file))
    label = feature_file.split("_")[0]
    new_size = len(features) - (len(features) % chunk_size)
    features = features[:new_size]
    chunks = torch.chunk(features, new_size // chunk_size)
    labels = [label] * len(chunks)
    all_chunks.extend(chunks)
    all_labels.extend(labels)

torch.save({"features": all_chunks, "labels": all_labels}, os.path.join(bags_dir, "mil_bags.pth"))