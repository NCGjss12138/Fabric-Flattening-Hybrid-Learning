import numpy as np
import pandas as pd
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.ndimage import label, center_of_mass

# ----------------------------
# UNet definition (same as training-time)
# ----------------------------
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
        nn.ReLU(inplace=True)
    )

class DeconvBlock(nn.Module):
    """Transpose-conv upsample + concat residual + conv_block."""
    def __init__(self, in_channels, residual_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv = conv_block(out_channels + residual_channels, out_channels)
    def forward(self, x, residual):
        x = self.deconv(x)
        x = torch.cat([x, residual], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    """UNet with candidate-map fusion."""
    def __init__(self, height=256, width=256, filters=16):
        super(UNet, self).__init__()
        self.height = height
        self.width = width
        # encoder (origin)
        self.conv1 = conv_block(1, filters)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = conv_block(filters, filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = conv_block(filters * 2, filters * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = conv_block(filters * 4, filters * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = conv_block(filters * 8, filters * 8)
        # candidate branch
        self.cand_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # decoder
        self.deconv6 = DeconvBlock(in_channels=filters * 8, residual_channels=filters * 8, out_channels=filters * 8)
        self.deconv7 = DeconvBlock(in_channels=filters * 8 + 1, residual_channels=filters * 4, out_channels=filters * 4)
        self.deconv8 = DeconvBlock(in_channels=filters * 4, residual_channels=filters * 2, out_channels=filters * 2)
        self.deconv9 = DeconvBlock(in_channels=filters * 2, residual_channels=filters, out_channels=filters)
        self.final_conv = nn.Conv2d(filters + 1, 1, kernel_size=1)
    def forward(self, origin, candidate):
        # encoder
        conv1 = self.conv1(origin)
        conv1_pool = self.pool1(conv1)
        conv2 = self.conv2(conv1_pool)
        conv2_pool = self.pool2(conv2)
        conv3 = self.conv3(conv2_pool)
        conv3_pool = self.pool3(conv3)
        conv4 = self.conv4(conv3_pool)
        conv4_pool = self.pool4(conv4)
        conv5 = self.conv5(conv4_pool)
        # candidate multiscale
        cand1 = self.cand_pool(candidate)
        cand2 = self.cand_pool(cand1)
        cand3 = self.cand_pool(cand2)
        cand4 = self.cand_pool(cand3)
        cand4_up = self.upsample(cand4)
        cand1_up = self.upsample(cand1)
        # decoder + fusion
        deconv6 = self.deconv6(conv5, conv4)
        deconv6 = torch.cat([deconv6, cand4_up], dim=1)
        deconv7 = self.deconv7(deconv6, conv3)
        deconv8 = self.deconv8(deconv7, conv2)
        deconv9 = self.deconv9(deconv8, conv1)
        deconv9 = torch.cat([deconv9, cand1_up], dim=1)
        output = self.final_conv(deconv9)
        return torch.sigmoid(output)

# ----------------------------
# Device and model loading
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_weights_path = '/data/user/models/unet_best.pth'
model = UNet(height=256, width=256, filters=16).to(device)
model.load_state_dict(torch.load(model_weights_path, map_location=device))
model.eval()

# ----------------------------
# Utilities
# ----------------------------
def load_image_cv2(path, target_size=(256, 256)):
    """Read grayscale image, resize to target_size, normalize to [0,1], return (H,W,1) float32 array."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found: {path}")
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def extract_centroids(image):
    """Find exactly two external contours and return their centroids sorted by x (left, right)."""
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 2:
        raise ValueError(f"Image should have exactly two regions, but found {len(contours)}.")
    cents = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cents.append((cX, cY))
        else:
            cents.append((0, 0))
    cents = sorted(cents, key=lambda x: x[0])
    return cents[0], cents[1]

def extract_top_two_centroids(mask):
    """Return centroids of the two largest connected components; use [0,0] if missing."""
    labeled_array, num_features = label(mask)
    if num_features == 0:
        return np.array([0, 0]), np.array([0, 0])
    region_sizes = [(i, np.sum(labeled_array == i)) for i in range(1, num_features + 1)]
    region_sizes = sorted(region_sizes, key=lambda x: x[1], reverse=True)
    largest_region = region_sizes[0][0]
    second_region = region_sizes[1][0] if len(region_sizes) > 1 else None
    largest_centroid = center_of_mass(mask, labeled_array, largest_region)
    second_centroid = center_of_mass(mask, labeled_array, second_region) if second_region is not None else np.array([0, 0])
    return np.array(largest_centroid), np.array(second_centroid)

def map_to_closest_candidate(centroid, candidate_image, side='left'):
    """Map a centroid (row, col) to the nearest candidate white pixel on the given side."""
    h, w = candidate_image.shape
    cx = w // 2
    if side == 'left':
        pts = np.column_stack(np.where(candidate_image[:, :cx] > 0.5))
    else:
        pts = np.column_stack(np.where(candidate_image[:, cx:] > 0.5))
        pts[:, 1] += cx
    if len(pts) == 0:
        return np.array([0, 0])
    dists = np.linalg.norm(pts - centroid, axis=1)
    return pts[np.argmin(dists)]

# ----------------------------
# Part 1: run UNet on training set to save y_pred and ground-truth points
# ----------------------------
origin_path = '/data/user/test/origin'
candidate_path = '/data/user/test/candidate'
output_path = '/data/user/test/output'              # ground-truth operation point image
y_pred_save_path = '/data/user/results/y_pred'
points_save_path = '/data/user/results/points'
os.makedirs(y_pred_save_path, exist_ok=True)
os.makedirs(points_save_path, exist_ok=True)

for filename in os.listdir(origin_path):
    if filename.lower().endswith('.bmp'):
        # load origin & candidate
        origin_img = load_image_cv2(os.path.join(origin_path, filename))
        candidate_img = load_image_cv2(os.path.join(candidate_path, filename))

        # to tensors (1,C,H,W)
        origin_tensor = torch.from_numpy(origin_img.transpose(2, 0, 1)).unsqueeze(0).to(device)
        candidate_tensor = torch.from_numpy(candidate_img.transpose(2, 0, 1)).unsqueeze(0).to(device)

        # inference
        with torch.no_grad():
            y_pred_tensor = model(origin_tensor, candidate_tensor)
        y_pred = y_pred_tensor.squeeze(0).cpu().numpy()  # shape (1,256,256)

        # save y_pred
        y_pred_filename = os.path.join(y_pred_save_path, filename.replace('.bmp', '_y_pred.npy'))
        np.save(y_pred_filename, y_pred)

        # load ground-truth operation-point image and extract two centroids
        output_img = cv2.imread(os.path.join(output_path, filename), cv2.IMREAD_GRAYSCALE)
        try:
            left_point, right_point = extract_centroids(output_img)
        except ValueError as e:
            print(f"Error processing {filename}: {e}")
            continue

        points_filename = os.path.join(points_save_path, filename.replace('.bmp', '_points.npy'))
        points_data = {'left': left_point, 'right': right_point}
        np.save(points_filename, points_data)
        print(f"Processed and saved for {filename}")

# ----------------------------
# Part 2: run on test set, extract candidate points and write overlays
# ----------------------------
origin_folder = '/data/user/test/origin'
candidate_folder = '/data/user/test/candidate'
test_image_files = [f for f in os.listdir(origin_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
test_pair = [(os.path.join(origin_folder, f), os.path.join(candidate_folder, f)) for f in test_image_files]

mask_folder = '/data/user/results/pred_mask'
output_folder = '/data/user/results/points_txt'
overlay_image_folder = '/data/user/results/overlay'
os.makedirs(mask_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(overlay_image_folder, exist_ok=True)

for origin_image_path, candidate_image_path in test_pair:
    # load origin & candidate
    origin_img = load_image_cv2(origin_image_path)
    candidate_img = load_image_cv2(candidate_image_path)

    # to tensors (1,C,H,W)
    origin_tensor = torch.from_numpy(origin_img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    candidate_tensor = torch.from_numpy(candidate_img.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # inference
    with torch.no_grad():
        predicted_mask_tensor = model(origin_tensor, candidate_tensor)
    predicted_mask = predicted_mask_tensor.squeeze(0).cpu().numpy()[0]  # (256,256)

    # save predicted mask as uint8 image
    image_filename = os.path.splitext(os.path.basename(origin_image_path))[0]
    predicted_mask_img = (predicted_mask * 255).astype(np.uint8)
    predicted_mask_path = os.path.join(mask_folder, f'{image_filename}_predicted_mask.bmp')
    cv2.imwrite(predicted_mask_path, predicted_mask_img)
    print(f"Predicted mask saved to {predicted_mask_path}")

    # threshold around max value
    maxpixel = np.max(predicted_mask)
    minpixel = maxpixel - 0.9
    binary_mask = np.where((predicted_mask >= minpixel) & (predicted_mask <= maxpixel), 1, 0)

    # split by midline
    h, w = binary_mask.shape
    cx = w // 2
    left_mask = binary_mask[:, :cx]
    right_mask = binary_mask[:, cx:]

    # top-2 components per side
    left_largest, left_second = extract_top_two_centroids(left_mask)
    right_largest, right_second = extract_top_two_centroids(right_mask)
    right_largest[1] += cx
    right_second[1] += cx

    # map to nearest candidate white pixel
    left_largest_point = map_to_closest_candidate(left_largest, candidate_img[:, :, 0], side='left')
    left_second_point = map_to_closest_candidate(left_second, candidate_img[:, :, 0], side='left')
    right_largest_point = map_to_closest_candidate(right_largest, candidate_img[:, :, 0], side='right')
    right_second_point = map_to_closest_candidate(right_second, candidate_img[:, :, 0], side='right')

    # write coords to txt
    output_txt_path = os.path.join(output_folder, f'{image_filename}_points.txt')
    with open(output_txt_path, 'w') as f:
        f.write(f'Left largest point: {left_largest_point}\n')
        f.write(f'Left second largest point: {left_second_point}\n')
        f.write(f'Right largest point: {right_largest_point}\n')
        f.write(f'Right second largest point: {right_second_point}\n')

    # draw overlay (points on original image)
    overlay = (origin_img[:, :, 0] * 255).astype(np.uint8)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    cv2.circle(overlay, tuple(left_largest_point[::-1].astype(int)), 6, (0, 255, 0), -1)
    cv2.circle(overlay, tuple(right_largest_point[::-1].astype(int)), 6, (0, 255, 0), -1)

    overlay_image_path = os.path.join(overlay_image_folder, f'{image_filename}.bmp')
    cv2.imwrite(overlay_image_path, overlay)
    print(f"Processed {image_filename}: points saved, overlay saved to {overlay_image_path}")
