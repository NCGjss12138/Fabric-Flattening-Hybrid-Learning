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

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using CUDA" if torch.cuda.is_available() else "Using CPU")


# ----------------------------
# Pair builder
# ----------------------------
def make_pair(origins, candidates, labels):
    pairs = []
    for origin, candidate, label in zip(origins, candidates, labels):
        pairs.append((origin, candidate, label))
    return pairs


# ----------------------------
# Paths
# ----------------------------
origin_path = Path("/data/user/train/origin/")
candidate_path = Path("/data/user/train/candidate/")
chosen_path = Path("/data/user/train/output/")

train_origins = list(origin_path.glob("*.bmp"))
train_candidates = list(candidate_path.glob("*.bmp"))
train_labels = list(chosen_path.glob("*.bmp"))

print(f"Number of origin images: {len(train_origins)}")
print(f"Number of candidate images: {len(train_candidates)}")
print(f"Number of label images: {len(train_labels)}")

assert len(train_origins) == len(train_candidates) == len(train_labels), "Mismatch in number of images"

train_pair = make_pair(train_origins, train_candidates, train_labels)
train_pair, val_pair = train_test_split(train_pair, test_size=0.16, random_state=42)


# ----------------------------
# Dataset (only H/V flips for augmentation)
# ----------------------------
class SegmentationDataset(Dataset):
    def __init__(self, pair, height=256, width=256, augment=True):
        self.pair = pair
        self.height = height
        self.width = width
        self.augment = augment

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, idx):
        origin_path, candidate_path, label_path = self.pair[idx]

        # load as grayscale
        origin = cv2.imread(str(origin_path), cv2.IMREAD_GRAYSCALE)
        candidate = cv2.imread(str(candidate_path), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)

        # resize to target size
        origin = cv2.resize(origin, (self.width, self.height))
        candidate = cv2.resize(candidate, (self.width, self.height))
        label = cv2.resize(label, (self.width, self.height))

        # add channel dimension
        origin = np.expand_dims(origin, axis=-1)
        candidate = np.expand_dims(candidate, axis=-1)
        label = np.expand_dims(label, axis=-1)

        # normalize to [0,1]
        origin = origin.astype(np.float32) / 255.0
        candidate = candidate.astype(np.float32) / 255.0
        label = label.astype(np.float32) / 255.0

        # augmentation: horizontal/vertical flip only
        if self.augment:
            if np.random.rand() < 0.5:
                origin = np.fliplr(origin)
                candidate = np.fliplr(candidate)
                label = np.fliplr(label)
            if np.random.rand() < 0.5:
                origin = np.flipud(origin)
                candidate = np.flipud(candidate)
                label = np.flipud(label)

        # ensure shape (H, W, 1)
        if origin.shape != (self.height, self.width, 1):
            origin = np.expand_dims(origin, axis=-1)
        if candidate.shape != (self.height, self.width, 1):
            candidate = np.expand_dims(candidate, axis=-1)
        if label.shape != (self.height, self.width, 1):
            label = np.expand_dims(label, axis=-1)

        # to tensor (C, H, W)
        origin = torch.from_numpy(origin.transpose(2, 0, 1).copy())
        candidate = torch.from_numpy(candidate.transpose(2, 0, 1).copy())
        label = torch.from_numpy(label.transpose(2, 0, 1).copy())
        return (origin, candidate), label


# ----------------------------
# UNet building blocks
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
    def __init__(self, in_channels, residual_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.conv = conv_block(out_channels + residual_channels, out_channels)

    def forward(self, x, residual):
        x = self.deconv(x)
        x = torch.cat([x, residual], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, height=256, width=256, filters=16):
        super(UNet, self).__init__()
        self.height = height
        self.width = width

        # encoder (origin branch)
        self.conv1 = conv_block(1, filters)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = conv_block(filters, filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = conv_block(filters * 2, filters * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = conv_block(filters * 4, filters * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = conv_block(filters * 8, filters * 8)

        # candidate branch (downsample then upsample)
        self.cand_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # decoder
        self.deconv6 = DeconvBlock(in_channels=filters * 8, residual_channels=filters * 8, out_channels=filters * 8)
        self.deconv7 = DeconvBlock(in_channels=filters * 8 + 1, residual_channels=filters * 4, out_channels=filters * 4)
        self.deconv8 = DeconvBlock(in_channels=filters * 4, residual_channels=filters * 2, out_channels=filters * 2)
        self.deconv9 = DeconvBlock(in_channels=filters * 2, residual_channels=filters, out_channels=filters)
        self.final_conv = nn.Conv2d(filters + 1, 1, kernel_size=1)

    def forward(self, origin, candidate):
        # encoder (origin)
        conv1 = self.conv1(origin)
        conv1_pool = self.pool1(conv1)
        conv2 = self.conv2(conv1_pool)
        conv2_pool = self.pool2(conv2)
        conv3 = self.conv3(conv2_pool)
        conv3_pool = self.pool3(conv3)
        conv4 = self.conv4(conv3_pool)
        conv4_pool = self.pool4(conv4)
        conv5 = self.conv5(conv4_pool)

        # candidate scales
        cand1 = self.cand_pool(candidate)
        cand2 = self.cand_pool(cand1)
        cand3 = self.cand_pool(cand2)
        cand4 = self.cand_pool(cand3)

        # upsample candidate features
        cand4_up = self.upsample(cand4)
        cand1_up = self.upsample(cand1)

        # decoder with fusion
        deconv6 = self.deconv6(conv5, conv4)
        deconv6 = torch.cat([deconv6, cand4_up], dim=1)
        deconv7 = self.deconv7(deconv6, conv3)
        deconv8 = self.deconv8(deconv7, conv2)
        deconv9 = self.deconv9(deconv8, conv1)
        deconv9 = torch.cat([deconv9, cand1_up], dim=1)

        output = self.final_conv(deconv9)
        output = torch.sigmoid(output)
        return output


# ----------------------------
# Losses
# ----------------------------
def dice_loss(y_true, y_pred, smooth=1):
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    intersection = (y_true_flat * y_pred_flat).sum()
    return 1 - (2. * intersection + smooth) / (y_true_flat.sum() + y_pred_flat.sum() + smooth)


bce_loss = nn.BCELoss()


def dice_bce_loss(y_true, y_pred):
    d_loss = dice_loss(y_true, y_pred)
    b_loss = bce_loss(y_pred, y_true)
    return 0.3 * d_loss + 0.7 * b_loss


mse_loss = nn.MSELoss()  # optional metric

# ----------------------------
# Training settings
# ----------------------------
height = 256
width = 256
epochs = 1000
batch_size = 64

train_dataset = SegmentationDataset(train_pair, height=height, width=width, augment=True)
val_dataset = SegmentationDataset(val_pair, height=height, width=width, augment=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ----------------------------
# Model / optimizer / scheduler
# ----------------------------
model = UNet(height=height, width=width, filters=16).to(device)
print(model)

torch.save(model.state_dict(), '/data/youchun/Pre-training/train/log/log_pytorch_2_14_37/unet(sigmoid).pth')

optimizer = optim.Adam(model.parameters(), lr=5e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, min_lr=5e-7, verbose=True)

best_val_loss = float('inf')
threshold = 0.25

# ----------------------------
# Train loop
# ----------------------------
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Training]", leave=False)

    for (inputs, targets) in train_loader_tqdm:
        origin, candidate = inputs
        origin = origin.to(device)
        candidate = candidate.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(origin, candidate)
        loss = dice_bce_loss(targets, outputs)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * origin.size(0)
        train_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}")

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Validation]", leave=False)

    with torch.no_grad():
        for (inputs, targets) in val_loader_tqdm:
            origin, candidate = inputs
            origin = origin.to(device)
            candidate = candidate.to(device)
            targets = targets.to(device)
            outputs = model(origin, candidate)
            loss = dice_bce_loss(targets, outputs)
            val_loss += loss.item() * origin.size(0)
            val_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}")

    val_loss /= len(val_loader.dataset)
    print(f"\nEpoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    scheduler.step(val_loss)

    if val_loss < threshold and val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = f"/data/user/train/log/top-weights(sigmoid)_{val_loss:.4f}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch + 1}: val_loss improved to {val_loss:.4f}, saving model to {save_path}")
