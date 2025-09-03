import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
from pytorch_wavelets import DWTForward  # pip install pytorch_wavelets
import random
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using CUDA" if torch.cuda.is_available() else "Using CPU")


# =======================================
# 1) Basic modules: HWD and CrossAttentionBlock
# =======================================
class HWD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, :, :]
        y_LH = yH[0][:, :, 1, :, :]
        y_HH = yH[0][:, :, 2, :, :]
        x_cat = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x_out = self.conv_bn_relu(x_cat)
        return x_out


class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pooling=True):
        super(CrossAttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pooling = pooling
        if pooling:
            self.pool = nn.MaxPool2d(2)
        self.alpha = nn.Parameter(torch.ones(1))
        self.hwd = HWD(1, 1)

    def forward(self, x, mask):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pooling:
            x = self.pool(x)
            mask = self.hwd(mask)
            x = x * (1 + self.alpha * mask)
        else:
            mask_down = self.hwd(mask)
            mask_down = F.interpolate(mask_down, size=x.shape[2:], mode='nearest')
            x = x * (1 + self.alpha * mask_down)
        return x, mask


# =======================================
# 2) Actor network
# =======================================
class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()
        self.initial_conv = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.initial_pool = nn.MaxPool2d(2)  # 256->128
        self.hwd_init = HWD(1, 1)
        self.block1 = CrossAttentionBlock(in_channels=32, out_channels=64, kernel_size=3, pooling=True)
        self.block2 = CrossAttentionBlock(in_channels=64, out_channels=128, kernel_size=3, pooling=True)
        self.block3 = CrossAttentionBlock(in_channels=128, out_channels=256, kernel_size=3, pooling=True)
        self.fc = nn.Sequential(
            nn.Linear(256 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 4),
            nn.Sigmoid()  # outputs in [0,1]
        )

    def forward(self, origin_img, y_pred, mask):
        origin_img = origin_img.to(device)
        y_pred = y_pred.to(device)
        mask = mask.to(device)
        x = torch.cat([origin_img, y_pred], dim=1)  # [B,2,256,256]
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.relu(x)
        x = self.initial_pool(x)  # [B,32,128,128]
        mask = self.hwd_init(mask)  # [B,1,128,128]
        x = x * (1 + mask)
        x, mask = self.block1(x, mask)
        x, mask = self.block2(x, mask)
        x, mask = self.block3(x, mask)
        x = x.view(x.size(0), -1)
        pred_points = self.fc(x)
        return pred_points


# =======================================
# 3) UNet
# =======================================
class UNet(nn.Module):
    def __init__(self, height=256, width=256, filters=16):
        super(UNet, self).__init__()
        self.height = height
        self.width = width
        self.conv1 = self.conv_block(1, filters)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = self.conv_block(filters, filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = self.conv_block(filters * 2, filters * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = self.conv_block(filters * 4, filters * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = self.conv_block(filters * 8, filters * 8)
        self.cand_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv6 = DeconvBlock(in_channels=filters * 8, residual_channels=filters * 8, out_channels=filters * 8)
        self.deconv7 = DeconvBlock(in_channels=filters * 8 + 1, residual_channels=filters * 4, out_channels=filters * 4)
        self.deconv8 = DeconvBlock(in_channels=filters * 4, residual_channels=filters * 2, out_channels=filters * 2)
        self.deconv9 = DeconvBlock(in_channels=filters * 2, residual_channels=filters, out_channels=filters)
        self.final_conv = nn.Conv2d(filters + 1, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
            nn.ReLU(inplace=True)
        )

    def forward(self, origin, candidate):
        conv1 = self.conv1(origin)
        conv1_pool = self.pool1(conv1)
        conv2 = self.conv2(conv1_pool)
        conv2_pool = self.pool2(conv2)
        conv3 = self.conv3(conv2_pool)
        conv3_pool = self.pool3(conv3)
        conv4 = self.conv4(conv3_pool)
        conv4_pool = self.pool4(conv4)
        conv5 = self.conv5(conv4_pool)
        cand1 = self.cand_pool(candidate)
        cand2 = self.cand_pool(cand1)
        cand3 = self.cand_pool(cand2)
        cand4 = self.cand_pool(cand3)
        cand4_up = self.upsample(cand4)
        cand1_up = self.upsample(cand1)
        deconv6 = self.deconv6(conv5, conv4)
        deconv6 = torch.cat([deconv6, cand4_up], dim=1)
        deconv7 = self.deconv7(deconv6, conv3)
        deconv8 = self.deconv8(deconv7, conv2)
        deconv9 = self.deconv9(deconv8, conv1)
        deconv9 = torch.cat([deconv9, cand1_up], dim=1)
        output = self.final_conv(deconv9)
        return torch.sigmoid(output)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, residual_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + residual_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, residual):
        x = self.deconv(x)
        x = torch.cat([x, residual], dim=1)
        x = self.conv(x)
        return x


# =======================================
# 4) Combined network = UNet + Actor
# =======================================
class CombinedNetwork(nn.Module):
    def __init__(self, unet, actor):
        super(CombinedNetwork, self).__init__()
        self.unet = unet
        self.actor = actor

    def forward(self, origin, candidate, mask):
        y_pred = self.unet(origin, candidate)
        pred_points = self.actor(origin, y_pred, mask)
        return y_pred, pred_points


# =======================================
# 5) Critic network (no global average pooling)
# =======================================
class CriticNetwork(nn.Module):
    def __init__(self, action_dim=4, action_embed_dim=128):
        super(CriticNetwork, self).__init__()
        self.initial_conv = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.initial_pool = nn.MaxPool2d(2)  # 256->128

        self.hwd_init = HWD(1, 1)
        self.block1 = CrossAttentionBlock(in_channels=32, out_channels=64, kernel_size=3, pooling=True)
        self.block2 = CrossAttentionBlock(in_channels=64, out_channels=128, kernel_size=3, pooling=True)
        self.block3 = CrossAttentionBlock(in_channels=128, out_channels=256, kernel_size=3, pooling=True)
        self.state_fc = nn.Sequential(
            nn.Linear(256 * 16 * 16, 128),
            nn.ReLU(inplace=True)
        )

        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, action_embed_dim),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 + action_embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()  # keep as-is
        )

    def forward(self, origin, y_pred, mask, action):
        # concatenate origin and y_pred as 2-channel state
        state_input = torch.cat([origin, y_pred], dim=1)  # [B,2,256,256]
        x = self.initial_conv(state_input)
        x = self.initial_bn(x)
        x = self.relu(x)
        x = self.initial_pool(x)  # [B,32,128,128]
        mask = self.hwd_init(mask)  # [B,1,128,128]
        x = x * (1 + mask)
        x, mask = self.block1(x, mask)
        x, mask = self.block2(x, mask)
        x, mask = self.block3(x, mask)
        x = x.view(x.size(0), -1)  # [B,256*16*16]
        state_feature = self.state_fc(x)  # [B,128]

        a_embed = self.action_embed(action)  # [B, action_embed_dim]
        x_cat = torch.cat([state_feature, a_embed], dim=1)  # [B,128 + action_embed_dim]
        q_value = self.fc(x_cat)  # [B,1]
        return q_value


# =======================================
# 6) Offline dataset: CombinedTrainingDataset
# =======================================
class CombinedTrainingDataset(Dataset):
    def __init__(self, root_dir, punish_file=None, transform=None, exploration=False):
        """
        exploration: if True, read actions from *_actor_coordinates.npy; else from *_points.npy.
        For utilization data: action file is 'img_{id}_points.npy';
        For exploration data: action file is 'img_{id}_actor_coordinates.npy'.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.sample_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.skipped = set()
        self.exploration = exploration
        if punish_file is not None:
            with open(punish_file, "r") as f:
                lines = f.readlines()
                self.skipped = set([line.strip() for line in lines])

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        sample_id = sample_dir.name  # e.g. "0001" (util) or "0001_2" (explore)

        # load origin/candidate/mask
        origin_path = sample_dir / f"img_{sample_id}.bmp"
        origin = cv2.imread(str(origin_path), cv2.IMREAD_GRAYSCALE)
        origin = cv2.resize(origin, (256, 256)).astype(np.float32) / 255.0

        candidate_path = sample_dir / f"img_{sample_id}_candidate.bmp"
        candidate = cv2.imread(str(candidate_path), cv2.IMREAD_GRAYSCALE)
        candidate = cv2.resize(candidate, (256, 256)).astype(np.float32) / 255.0

        mask_path = sample_dir / f"img_{sample_id}_mask.bmp"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256)).astype(np.float32) / 255.0

        # choose action file by exploration flag
        if self.exploration:
            action_path = sample_dir / f"img_{sample_id}_actor_coordinates.npy"
        else:
            action_path = sample_dir / f"img_{sample_id}_points.npy"
        points_dict = np.load(str(action_path), allow_pickle=True).item()
        left = np.array(points_dict['left'], dtype=np.float32)
        right = np.array(points_dict['right'], dtype=np.float32)
        target_points = np.concatenate([left, right], axis=0)

        # normalize by [256, 256] for both points
        img_width, img_height = 256, 256
        target_points[:2] = target_points[:2] / np.array([img_width, img_height], dtype=np.float32)
        target_points[2:] = target_points[2:] / np.array([img_width, img_height], dtype=np.float32)
        target_points = torch.tensor(target_points, dtype=torch.float32)

        # load reward, apply log scaling; apply penalty if in skipped list
        excel_path = sample_dir / f"img_{sample_id}.xlsx"
        df = pd.read_excel(str(excel_path))
        raw_reward = df["Value"].iloc[0]
        reward_val = np.log(raw_reward + 1) / np.log(101)
        reward = torch.tensor(reward_val, dtype=torch.float32)
        punishment = 3.0
        if sample_id in self.skipped:
            reward = reward - punishment

        if self.transform is not None:
            origin = self.transform(origin)
            candidate = self.transform(candidate)
            mask = self.transform(mask)

        origin = np.expand_dims(origin, axis=0)
        candidate = np.expand_dims(candidate, axis=0)
        mask = np.expand_dims(mask, axis=0)
        is_exploration = 1 if self.exploration else 0
        return origin, candidate, mask, target_points, reward, is_exploration


# =======================================
# 7) DataLoaders: utilization and exploration
# =======================================
util_dataset = CombinedTrainingDataset(
    root_dir="/data/user/combined_dataset/",
    punish_file="/data/user/combined_dataset/skipped_images.txt",
    exploration=False)
explore_dataset = CombinedTrainingDataset(
    root_dir="/data/user/combined_dataset_explore/",
    punish_file="/data/user/combined_dataset_explore/skipped_images.txt",
    exploration=True)

util_loader = DataLoader(util_dataset, batch_size=32, shuffle=True, num_workers=4)
explore_loader = DataLoader(explore_dataset, batch_size=32, shuffle=True, num_workers=4)

# =======================================
# 8) Load pretrained weights and build nets
# =======================================
unet = UNet(height=256, width=256, filters=16).to(device)
unet.load_state_dict(
    torch.load("/data/user/RL_Train/unet_best.pth"))
actor = ActorNetwork().to(device)
actor.load_state_dict(
    torch.load("/data/user/RL_Train/actor_best.pth"))

combined_net = CombinedNetwork(unet, actor).to(device)

# freeze CombinedNetwork (for critic pretrain)
for param in combined_net.unet.parameters():
    param.requires_grad = False
for param in combined_net.actor.parameters():
    param.requires_grad = False

critic_net = CriticNetwork(action_dim=4, action_embed_dim=128).to(device)
target_combined = CombinedNetwork(unet, actor).to(device)
target_critic = CriticNetwork(action_dim=4, action_embed_dim=128).to(device)
target_combined.load_state_dict(combined_net.state_dict())
target_critic.load_state_dict(critic_net.state_dict())

# optimizers
critic_optimizer = optim.Adam(critic_net.parameters(), lr=1e-4)
actor_optimizer = optim.Adam(
    list(combined_net.unet.parameters()) + list(combined_net.actor.parameters()), lr=1e-4)

tau = 0.005
file_name = "RL_5_second_round_3.0"
best_loss = float('inf')

# =======================================
# Phase 1: pretrain critic with exploration data
# =======================================
n_pretrain_epochs = 50
print("阶段一：预训练 Critic 网络")
for epoch in range(n_pretrain_epochs):
    combined_net.eval()
    critic_net.train()
    epoch_loss = 0.0
    for batch in tqdm(explore_loader, desc=f"Pretrain Critic Epoch {epoch + 1}/{n_pretrain_epochs}"):
        origin, candidate, mask, target_points, reward, _ = batch
        origin = origin.to(device)
        candidate = candidate.to(device)
        mask = mask.to(device)
        reward = reward.to(device).unsqueeze(1)

        with torch.no_grad():
            y_pred = combined_net.unet(origin, candidate)
        action = target_points.to(device)

        current_q = critic_net(origin, y_pred, mask, action)
        target_q = reward
        critic_loss = F.mse_loss(current_q, target_q)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        epoch_loss += critic_loss.item()
    avg_pretrain_loss = epoch_loss / len(explore_loader)
    print(f"Pretrain Epoch {epoch + 1}/{n_pretrain_epochs}, Critic Loss: {avg_pretrain_loss:.6f}")

# =======================================
# Phase 2: joint training with utilization data
# =======================================
combined_net = CombinedNetwork(unet, actor).to(device)
for param in unet.parameters():
    param.requires_grad = False
for param in unet.final_conv.parameters():
    param.requires_grad = False
for param in combined_net.actor.parameters():
    param.requires_grad = True

n_joint_epochs = 100
print("阶段二：联合训练 (利用数据)")
for epoch in range(n_joint_epochs):
    combined_net.train()
    critic_net.train()
    epoch_loss = 0.0
    total_imitation_loss = 0.0
    total_rl_loss = 0.0
    total_critic_loss = 0.0
    num_batches = 0

    for batch in tqdm(util_loader, desc=f"Joint Training Epoch {epoch + 1}/{n_joint_epochs}"):
        origin, candidate, mask, target_points, reward, _ = batch
        origin = origin.to(device)
        candidate = candidate.to(device)
        mask = mask.to(device)
        target_points = target_points.to(device)
        reward = reward.to(device).unsqueeze(1)

        # critic update
        y_pred_critic = combined_net.unet(origin, candidate)
        actor_action_critic = combined_net.actor(origin, y_pred_critic, mask)
        current_q = critic_net(origin, y_pred_critic, mask, actor_action_critic)
        target_q = reward
        critic_loss = F.mse_loss(current_q, target_q)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # actor update
        y_pred_actor = combined_net.unet(origin, candidate)
        actor_action_actor = combined_net.actor(origin, y_pred_actor, mask)
        imitation_loss = F.mse_loss(actor_action_actor, target_points)
        rl_loss = -critic_net(origin, y_pred_actor, mask, actor_action_actor).mean()
        loss_actor = imitation_loss + rl_loss

        actor_optimizer.zero_grad()
        loss_actor.backward()
        actor_optimizer.step()

        # soft update targets
        for target_param, param in zip(target_critic.parameters(), critic_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(target_combined.parameters(), combined_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        batch_loss = critic_loss.item() + loss_actor.item()
        epoch_loss += batch_loss
        total_imitation_loss += imitation_loss.item()
        total_rl_loss += rl_loss.item()
        total_critic_loss += critic_loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    avg_imitation_loss = total_imitation_loss / (len(util_loader) if len(util_loader) > 0 else 1)
    avg_rl_loss = total_rl_loss / num_batches
    avg_critic_loss = total_critic_loss / num_batches

    print(f"Joint Training Epoch {epoch + 1}/{n_joint_epochs}, Avg Total Loss: {avg_loss:.6f}, "
          f"Imitation Loss: {avg_imitation_loss:.6f}, RL Loss: {avg_rl_loss:.6f}, "
          f"Critic Loss: {avg_critic_loss:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_dir = os.path.join("/data/user/RL_Train/", file_name, "best")
        os.makedirs(best_dir, exist_ok=True)
        torch.save(combined_net.state_dict(), os.path.join(best_dir, "combined_net.pth"))
        torch.save(unet.state_dict(), os.path.join(best_dir, "unet.pth"))
        torch.save(actor.state_dict(), os.path.join(best_dir, "actor.pth"))
        torch.save(critic_net.state_dict(), os.path.join(best_dir, "critic.pth"))
        print("Saved improved model in best folder!")

    # save checkpoints every 5 epochs (kept as original print text)
    if (epoch + 1) % 5 == 0:
        epoch_dir = os.path.join("/data/user/RL_Train/", file_name, str(epoch + 1))
        os.makedirs(epoch_dir, exist_ok=True)
        torch.save(combined_net.state_dict(), os.path.join(epoch_dir, "combined_net.pth"))
        torch.save(unet.state_dict(), os.path.join(epoch_dir, "unet.pth"))
        torch.save(actor.state_dict(), os.path.join(epoch_dir, "actor.pth"))
        torch.save(critic_net.state_dict(), os.path.join(epoch_dir, "critic.pth"))
        print(f"Saved model checkpoints for epoch {epoch + 1} in folder '{epoch + 1}'!")
