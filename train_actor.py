import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import torch.nn.functional as F
from tqdm import tqdm
import cv2

# requires pytorch_wavelets (pip install pytorch_wavelets)
from pytorch_wavelets import DWTForward

# =========================
# paths (edit to your environment)
# =========================
origin_path = '/data/user/train/origin'
y_pred_path = '/data/user/train/y_pred'
points_path = '/data/user/train/points_data'
mask_path = '/data/user/train/mask'


# =========================
# helpers
# =========================
def load_image(path):
    """Load grayscale image and scale to [0,1]."""
    image = load_img(path, color_mode="grayscale")
    image = img_to_array(image) / 255.0
    return np.squeeze(image, axis=-1)


def extract_y_pred_centroids(y_pred):
    """Extract two centroids from a binary map; return None if invalid."""
    # treat all-ones as invalid
    if np.all(y_pred > 0.5):
        return None
    _, binary_image = cv2.threshold(y_pred, 0.5, 1, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        return None
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
    if len(centroids) < 2:
        return None
    centroids = sorted(centroids, key=lambda x: x[0])
    return centroids[0], centroids[1]


# =========================
# Dataset
# =========================
class ActorDataset(Dataset):
    def __init__(self, origin_path, y_pred_path, points_path, mask_path):
        self.origin_path = origin_path
        self.y_pred_path = y_pred_path
        self.points_path = points_path
        self.mask_path = mask_path
        self.file_list = [f for f in os.listdir(origin_path) if f.endswith('.bmp')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        # origin image (grayscale)
        origin_img = load_image(os.path.join(self.origin_path, filename))
        # binary mask
        mask = load_image(os.path.join(self.mask_path, filename))
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1,256,256]
        # y_pred (raw probability map)
        y_pred = np.load(os.path.join(self.y_pred_path, filename.replace('.bmp', '_y_pred.npy')))
        y_pred = np.squeeze(y_pred)
        # ground-truth points (left/right)
        points = np.load(os.path.join(self.points_path, filename.replace('.bmp', '_points.npy')),
                         allow_pickle=True).item()
        left_point = torch.tensor(points['left'], dtype=torch.float32)
        right_point = torch.tensor(points['right'], dtype=torch.float32)
        # normalize points to [0,1] by image size
        img_height, img_width = 256, 256
        left_point /= torch.tensor([img_width, img_height], dtype=torch.float32)
        right_point /= torch.tensor([img_width, img_height], dtype=torch.float32)
        # to tensors with channel dim
        origin_img = torch.tensor(origin_img, dtype=torch.float32).unsqueeze(0)  # [1,256,256]
        y_pred = torch.tensor(y_pred, dtype=torch.float32).unsqueeze(0)  # [1,256,256]
        return origin_img, y_pred, left_point, right_point, mask


def collate_fn(batch):
    """Filter out None and stack."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    origin_imgs, y_preds, left_points, right_points, masks = zip(*batch)
    origin_imgs = torch.stack(origin_imgs)
    y_preds = torch.stack(y_preds)
    left_points = torch.stack(left_points)
    right_points = torch.stack(right_points)
    masks = torch.stack(masks)
    return origin_imgs, y_preds, left_points, right_points, masks


# =========================
# HWD module
# =========================
class HWD(nn.Module):
    """Haar Wavelet Downsampling block with 1-level DWT."""

    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: [B, in_ch, H, W]
        yL, yH = self.wt(x)  # yL: [B,in_ch,H/2,W/2], yH[0]: [B,in_ch,3,H/2,W/2]
        y_HL = yH[0][:, :, 0, :, :]
        y_LH = yH[0][:, :, 1, :, :]
        y_HH = yH[0][:, :, 2, :, :]
        x_cat = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)  # [B, in_ch*4, H/2, W/2]
        x_out = self.conv_bn_relu(x_cat)  # [B, out_ch,  H/2, W/2]
        return x_out


# =========================
# CrossAttentionBlock
# =========================
class CrossAttentionBlock(nn.Module):
    """
    Convolution + BN + ReLU, then reweight features using a downsampled mask (via HWD).
    If pooling=True, apply MaxPool(2) to x and HWD to mask; else downsample mask by HWD and upsample to x size.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, pooling=True):
        super(CrossAttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pooling = pooling
        if pooling:
            self.pool = nn.MaxPool2d(2)
        self.alpha = nn.Parameter(torch.ones(1))  # learnable scaling for mask
        self.hwd = HWD(1, 1)  # mask downsampling

    def forward(self, x, mask):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pooling:
            x = self.pool(x)  # halve spatial size
            mask = self.hwd(mask)  # halve mask via HWD
            x = x * (1 + self.alpha * mask)
        else:
            mask_down = self.hwd(mask)
            mask_down = F.interpolate(mask_down, size=x.shape[2:], mode='nearest')
            x = x * (1 + self.alpha * mask_down)
        return x, mask


# =========================
# Actor Network
# =========================
class ActorNetwork(nn.Module):
    """
    Concatenate image and y_pred as 2-channel input.
    Use HWD to downsample mask initially; then pass through three CrossAttentionBlocks.
    Keep full feature map (no GAP); flatten and use FC to output 4 normalized coords.
    """

    def __init__(self):
        super(ActorNetwork, self).__init__()
        self.initial_conv = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.initial_pool = nn.MaxPool2d(2)  # 256 -> 128

        # initial mask downsampling via HWD (factor 2)
        self.hwd_init = HWD(1, 1)  # [B,1,256,256] -> [B,1,128,128]

        # three CrossAttentionBlocks
        self.block1 = CrossAttentionBlock(in_channels=32, out_channels=64, kernel_size=3, pooling=True)  # 128 -> 64
        self.block2 = CrossAttentionBlock(in_channels=64, out_channels=128, kernel_size=3, pooling=True)  # 64 -> 32
        self.block3 = CrossAttentionBlock(in_channels=128, out_channels=256, kernel_size=3, pooling=True)  # 32 -> 16

        # FC head (flatten 256x16x16 -> 4)
        self.fc = nn.Sequential(
            nn.Linear(256 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 4),
            nn.Sigmoid()  # normalize outputs to [0,1]
        )

    def forward(self, origin_img, y_pred, mask):
        device = next(self.parameters()).device
        origin_img = origin_img.to(device)
        y_pred = y_pred.to(device)
        mask = mask.to(device)

        # concat image and y_pred -> [B,2,H,W]
        x = torch.cat([origin_img, y_pred], dim=1)
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.relu(x)
        x = self.initial_pool(x)  # [B,32,128,128]

        # downsample mask to [B,1,128,128] via HWD
        mask = self.hwd_init(mask)
        x = x * (1 + mask)

        # three attention blocks
        x, mask = self.block1(x, mask)  # [B,64,64,64]
        x, mask = self.block2(x, mask)  # [B,128,32,32]
        x, mask = self.block3(x, mask)  # [B,256,16,16]

        # flatten and predict 4 values
        x = x.view(x.size(0), -1)  # [B, 256*16*16]
        pred_points = self.fc(x)  # [B,4]
        return pred_points


# =========================
# Projection loss
# =========================
def projection_loss_with_order(pred_points, gt_left, gt_right, lambda_penalty=2.0, lambda_order=1.0, lambda_reward=0.5):
    """
    Encourage projected predicted points to stay within GT span and maintain left/right order.
    pred_points: (B,4) -> [Plx, Ply, Prx, Pry] in [0,1]
    gt_left, gt_right: (B,2) in [0,1]
    """
    P_left = pred_points[:, :2]
    P_right = pred_points[:, 2:]

    O = (gt_left + gt_right) / 2.0
    v = gt_right - gt_left
    v_norm_sq = torch.sum(v * v, dim=1, keepdim=True)

    t_left = torch.sum((P_left - gt_left) * v, dim=1, keepdim=True) / (v_norm_sq + 1e-8)
    t_right = torch.sum((P_right - gt_left) * v, dim=1, keepdim=True) / (v_norm_sq + 1e-8)

    P_left_proj = gt_left + t_left * v
    P_right_proj = gt_left + t_right * v

    d_pred_left = torch.norm(P_left_proj - O, dim=1)
    d_pred_right = torch.norm(P_right_proj - O, dim=1)

    d_gt_left = torch.norm(gt_left - O, dim=1)
    d_gt_right = torch.norm(gt_right - O, dim=1)

    # reward if inside span (smaller than GT distance), penalize if outside
    loss_left = torch.where(d_pred_left <= d_gt_left, (d_gt_left - d_pred_left) * lambda_reward,
                            (d_pred_left - d_gt_left) * lambda_penalty)
    loss_right = torch.where(d_pred_right <= d_gt_right, (d_gt_right - d_pred_right) * lambda_reward,
                             (d_pred_right - d_gt_right) * lambda_penalty)
    loss_proj = loss_left + loss_right

    # order constraint: prefer t_left ~ 0 and t_right ~ 1
    loss_order = torch.mean(torch.abs(t_left - 0.0)) + torch.mean(torch.abs(t_right - 1.0))

    return loss_proj.mean() + lambda_order * loss_order


# =========================
# Training
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor_network = ActorNetwork().to(device)
optimizer = optim.Adam(actor_network.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=8, verbose=True)
mse_loss = nn.MSELoss()

projection_loss_weight = 0.1
projection_loss_delay = 30

# data loaders
dataset = ActorDataset(origin_path, y_pred_path, points_path, mask_path)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

best_model_path = "/data/youchun/Pre-training/train_new/train_6/best_actor_model_M4_NewUnet_30_3_1.5.pth"
best_val_loss = float('inf')
num_epochs = 1000

for epoch in range(num_epochs):
    actor_network.train()
    epoch_loss = 0.0
    epoch_loss_mse = 0.0
    epoch_loss_proj = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        if batch is None:
            continue
        origin_img, y_pred, left_point, right_point, mask = batch
        origin_img = origin_img.to(device)
        y_pred = y_pred.to(device)
        mask = mask.to(device)
        left_point = left_point.to(device)
        right_point = right_point.to(device)
        target_points = torch.cat((left_point, right_point), dim=1)  # [B,4]

        optimizer.zero_grad()
        pred_points = actor_network(origin_img, y_pred, mask)
        loss_mse = mse_loss(pred_points, target_points)

        if epoch >= projection_loss_delay:
            loss_proj = projection_loss_with_order(
                pred_points, left_point, right_point,
                lambda_penalty=1.5, lambda_order=1.0, lambda_reward=1.0
            )
            loss = loss_mse + projection_loss_weight * loss_proj
            epoch_loss_proj += loss_proj.item()
        else:
            loss = loss_mse
            loss_proj = torch.tensor(0.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_loss_mse += loss_mse.item()

    avg_train_loss = epoch_loss / len(train_loader)
    avg_loss_mse = epoch_loss_mse / len(train_loader)
    avg_loss_proj = epoch_loss_proj / len(train_loader) if epoch >= projection_loss_delay else 0.0
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, "
          f"loss_mse: {avg_loss_mse:.4f}, loss_proj: {avg_loss_proj:.4f}")

    actor_network.eval()
    val_loss = 0.0
    val_loss_mse = 0.0
    val_loss_proj = 0.0
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            origin_img, y_pred, left_point, right_point, mask = batch
            origin_img = origin_img.to(device)
            y_pred = y_pred.to(device)
            mask = mask.to(device)
            left_point = left_point.to(device)
            right_point = right_point.to(device)
            target_points = torch.cat((left_point, right_point), dim=1)
            pred_points = actor_network(origin_img, y_pred, mask)
            if epoch >= projection_loss_delay:
                loss_proj = projection_loss_with_order(
                    pred_points, left_point, right_point,
                    lambda_penalty=1.5, lambda_order=1.0, lambda_reward=1.0
                )
                loss = loss_mse + projection_loss_weight * loss_proj
                val_loss_proj += loss_proj.item()
            else:
                loss = loss_mse
                loss_proj = torch.tensor(0.0)

            val_loss += loss.item()
            val_loss_mse += loss_mse.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_loss_mse = val_loss_mse / len(val_loader)
    avg_val_loss_proj = val_loss_proj / len(val_loader) if epoch >= projection_loss_delay else 0.0
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, "
          f"loss_mse: {avg_val_loss_mse:.4f}, loss_proj: {avg_val_loss_proj:.4f}")
    scheduler.step(avg_val_loss)

    if avg_val_loss_mse < best_val_loss:
        best_val_loss = avg_val_loss_mse
        torch.save(actor_network.state_dict(), best_model_path)
        print(f"New best model saved with Validation Loss: {best_val_loss:.4f}")
