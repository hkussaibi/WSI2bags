import numpy as np
import cv2
from tqdm import tqdm

def RGB2HSD(X):
    eps = np.finfo(float).eps
    X[X == 0.0] = eps
    OD = -np.log(X / 1.0)
    D = np.mean(OD, axis=3)
    D[D == 0.0] = eps
    cx = OD[:, :, :, 0] / D - 1.0
    cy = (OD[:, :, :, 1] - OD[:, :, :, 2]) / (np.sqrt(3.0) * D)
    D = np.expand_dims(D, 3)
    cx = np.expand_dims(cx, 3)
    cy = np.expand_dims(cy, 3)
    X_HSD = np.concatenate((D, cx, cy), 3)
    return X_HSD

def clean_thumbnail(thumbnail):
    thumbnail_arr = np.asarray(thumbnail)
    wthumbnail = np.zeros_like(thumbnail_arr)
    wthumbnail[:, :, :] = thumbnail_arr[:, :, :]
    thumbnail_std = np.std(wthumbnail, axis=2)
    wthumbnail[thumbnail_std < 5] = 255
    thumbnail_HSD = RGB2HSD(np.array([wthumbnail.astype('float32') / 255.]))[0]
    kernel = np.ones((30, 30), np.float32) / 900
    thumbnail_HSD_mean = cv2.filter2D(thumbnail_HSD[:, :, 2], -1, kernel)
    wthumbnail[thumbnail_HSD_mean < 0.05] = 255
    return wthumbnail

def get_tissue_mask(thumbnail):
    cthumbnail = clean_thumbnail(thumbnail)
    tissue_mask = (cthumbnail.mean(axis=2) != 255) * 1
    return tissue_mask

def get_valid_patches(slide, thumbnail, tissue_threshold=0.9, patch_size=250):
    tissue_mask = get_tissue_mask(thumbnail)
    w, h = slide.dimensions
    mask_hratio = (tissue_mask.shape[0] / h) * patch_size
    mask_wratio = (tissue_mask.shape[1] / w) * patch_size
    valid_patches = []
    for i, hi in enumerate(range(0, h, patch_size)):
        for j, wi in enumerate(range(0, w, patch_size)):
            mi = int(i * mask_hratio)
            mj = int(j * mask_wratio)
            patch_mask = tissue_mask[mi:mi + int(mask_hratio), mj:mj + int(mask_wratio)]
            tissue_coverage = np.count_nonzero(patch_mask) / patch_mask.size
            if tissue_coverage >= tissue_threshold:
                patch_region = slide.read_region((wi, hi), 0, (patch_size, patch_size)).convert('RGB')
                valid_patches.append(patch_region)
    return valid_patches