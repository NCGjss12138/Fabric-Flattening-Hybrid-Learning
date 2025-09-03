import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from pytorch_wavelets import DWTForward
from tqdm import tqdm

# ---------------------------
# 设备设置
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# 网络结构定义
# ---------------------------
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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
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

class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()
        # 初始卷积层：将原图和 UNet 的 y_pred 拼接为2通道输入
        self.initial_conv = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.initial_pool = nn.MaxPool2d(2)  # 256 -> 128

        # 使用 HWD 对 mask 下采样
        self.hwd_init = HWD(1, 1)  # 输入mask：[B,1,256,256] -> 输出：[B,1,128,128]
        # 三个 CrossAttentionBlock 模块
        self.block1 = CrossAttentionBlock(in_channels=32, out_channels=64, kernel_size=3, pooling=True)   # 128 -> 64
        self.block2 = CrossAttentionBlock(in_channels=64, out_channels=128, kernel_size=3, pooling=True)  # 64 -> 32
        self.block3 = CrossAttentionBlock(in_channels=128, out_channels=256, kernel_size=3, pooling=True) # 32 -> 16

        # 全连接层：输入尺寸 256*16*16，输出4个坐标（归一化到[0,1]）
        self.fc = nn.Sequential(
            nn.Linear(256 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )
    def forward(self, origin_img, y_pred, mask):
        # 将 origin_img 和 y_pred 拼接成 [B,2,256,256]
        x = torch.cat([origin_img, y_pred], dim=1)
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.relu(x)
        x = self.initial_pool(x)  # 输出尺寸 [B,32,128,128]
        mask = self.hwd_init(mask)  # mask 下采样到 [B,1,128,128]
        x = x * (1 + mask)
        x, mask = self.block1(x, mask)  # 输出 [B,64,64,64]
        x, mask = self.block2(x, mask)  # 输出 [B,128,32,32]
        x, mask = self.block3(x, mask)  # 输出 [B,256,16,16]
        x = x.view(x.size(0), -1)
        pred_points = self.fc(x)
        return pred_points

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
        conv1 = self.conv1(origin)           # [B, filters, 256,256]
        conv1_pool = self.pool1(conv1)         # [B, filters,128,128]
        conv2 = self.conv2(conv1_pool)         # [B, filters*2,128,128]
        conv2_pool = self.pool2(conv2)         # [B, filters*2,64,64]
        conv3 = self.conv3(conv2_pool)         # [B, filters*4,64,64]
        conv3_pool = self.pool3(conv3)         # [B, filters*4,32,32]
        conv4 = self.conv4(conv3_pool)         # [B, filters*8,32,32]
        conv4_pool = self.pool4(conv4)         # [B, filters*8,16,16]
        conv5 = self.conv5(conv4_pool)         # [B, filters*8,16,16]
        cand1 = self.cand_pool(candidate)      # [B,1,128,128]
        cand2 = self.cand_pool(cand1)          # [B,1,64,64]
        cand3 = self.cand_pool(cand2)          # [B,1,32,32]
        cand4 = self.cand_pool(cand3)          # [B,1,16,16]
        cand4_up = self.upsample(cand4)         # [B,1,32,32]
        cand1_up = self.upsample(cand1)         # [B,1,256,256]
        deconv6 = self.deconv6(conv5, conv4)    # [B, filters*8,32,32]
        deconv6 = torch.cat([deconv6, cand4_up], dim=1)  # [B, filters*8+1,32,32]
        deconv7 = self.deconv7(deconv6, conv3)  # [B, filters*4,64,64]
        deconv8 = self.deconv8(deconv7, conv2)  # [B, filters*2,128,128]
        deconv9 = self.deconv9(deconv8, conv1)  # [B, filters,256,256]
        deconv9 = torch.cat([deconv9, cand1_up], dim=1)  # [B, filters+1,256,256]
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

class CombinedNetwork(nn.Module):
    def __init__(self, unet, actor):
        super(CombinedNetwork, self).__init__()
        self.unet = unet
        self.actor = actor
    def forward(self, origin, candidate, mask):
        y_pred = self.unet(origin, candidate)
        pred_points = self.actor(origin, y_pred, mask)
        return y_pred, pred_points

# ---------------------------
# 下面定义一个测试数据集（假设你的测试数据采用与训练相同的文件夹结构，每个子文件夹包含：img_{id}.bmp, img_{id}_candidate.bmp, img_{id}_mask.bmp）
# ---------------------------
class TestDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        # 假设每个子文件夹对应一个样本
        self.sample_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
    def __len__(self):
        return len(self.sample_dirs)
    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        sample_id = sample_dir.name  # 如 "0000"
        # 加载原图
        origin_path = sample_dir / f"img_{sample_id}.bmp"
        origin = cv2.imread(str(origin_path), cv2.IMREAD_GRAYSCALE)
        origin = cv2.resize(origin, (256,256)).astype(np.float32)/255.0
        # 加载 candidate 图（与原图类似）
        candidate_path = sample_dir / f"img_{sample_id}_candidate.bmp"
        candidate = cv2.imread(str(candidate_path), cv2.IMREAD_GRAYSCALE)
        candidate = cv2.resize(candidate, (256,256)).astype(np.float32)/255.0
        # 加载 mask
        mask_path = sample_dir / f"img_{sample_id}_mask.bmp"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256,256)).astype(np.float32)/255.0

        # 转换为 torch tensor
        origin = torch.tensor(origin, dtype=torch.float32).unsqueeze(0)  # [1,256,256]
        candidate = torch.tensor(candidate, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return origin, candidate, mask, sample_id

# ---------------------------
# 定义输出路径
# ---------------------------
file_name = "RL_5_second_round_2.0"
file_number = "20"
output_name = "output_20"
output_coordinate_path = "/data/youchun/Pre-training/RL_Train/" + file_name + "/" +  output_name + "/output_coordinate/"
output_image_path = "/data/youchun/Pre-training/RL_Train/" + file_name + "/" +  output_name + "/output_image/"
overlay_output_path = "/data/youchun/Pre-training/RL_Train/" + file_name + "/" +  output_name + "/overlay_output/"

os.makedirs(output_coordinate_path, exist_ok=True)
os.makedirs(output_image_path, exist_ok=True)
os.makedirs(overlay_output_path, exist_ok=True)

# ---------------------------
# 加载训练好的联合网络模型（CombinedNetwork），UNet和Actor已经集成
# ---------------------------
# 请修改下面的路径为你的保存路径
combined_net = CombinedNetwork(UNet(height=256, width=256, filters=16), ActorNetwork()).to(device)
#unet = UNet(height=256, width=256, filters=16).to(device)
#unet.load_state_dict(
    #torch.load("/data/youchun/Pre-training/UNet/log/log_pytorch_2_14_37/top-weights(sigmoid)_0.0544.pth"))
#actor = ActorNetwork().to(device)
#actor.load_state_dict(
    #torch.load("/data/youchun/Pre-training/train_new/train_6/best_actor_model_M4_NewUnet_30_3_2.5.pth"))
combined_net.load_state_dict(torch.load("/data/youchun/Pre-training/RL_Train/" + file_name + "/" +  file_number + "/combined_net.pth", map_location=device))
#combined_net = CombinedNetwork(unet, actor).to(device)
combined_net.eval()

# ---------------------------
# 构造测试数据集和 DataLoader
# ---------------------------
test_root = "/data/youchun/Pre-training/RL_Train/first_round/CombinedTraining_dataset/"
test_dataset = TestDataset(test_root)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ---------------------------
# 测试并保存结果
# ---------------------------
with torch.no_grad():
    for origin_img, candidate_img, mask, sample_id in tqdm(test_loader, desc="Testing"):
        origin_img = origin_img.to(device)
        candidate_img = candidate_img.to(device)
        mask = mask.to(device)
        y_pred, pred_points = combined_net(origin_img, candidate_img, mask)
        # pred_points 的值为归一化坐标 [0,1]
        pred_points_np = pred_points.cpu().numpy()[0]  # [4]
        # 分别提取左点和右点
        left_point = pred_points_np[:2]
        right_point = pred_points_np[2:]
        # 转换为像素坐标
        left_point = np.clip((left_point * 256).astype(int), 0, 255)
        right_point = np.clip((right_point * 256).astype(int), 0, 255)
        print(f"Sample {sample_id[0]} predicted points (normalized): {pred_points_np}")
        print(f"Predicted left point: {left_point}, right point: {right_point}")
        print(f"Real Predicted left point: {left_point} * 256, right point: {right_point} * 256")
        
        # 保存坐标到 npy 文件
        coord_filename = os.path.join(output_coordinate_path, sample_id[0] + "_actor_coordinates.npy")
        coordinates = {'left': tuple(left_point), 'right': tuple(right_point)}
        np.save(coord_filename, coordinates)
        
        # 创建一张空白图像用于绘制点（彩色图）
        img_canvas = np.zeros((256,256,3), dtype=np.uint8)
        cv2.circle(img_canvas, tuple(left_point), radius=4, color=(0,255,0), thickness=-1)
        cv2.circle(img_canvas, tuple(right_point), radius=4, color=(0,0,255), thickness=-1)
        image_output_path = os.path.join(output_image_path, sample_id[0] + "_actor_points.png")
        cv2.imwrite(image_output_path, img_canvas)
        
        # 读取原图（彩色），在上面叠加预测点
        origin_img_np = cv2.imread(os.path.join(str(test_dataset.root_dir / sample_id[0]), f"img_{sample_id[0]}.bmp"))
        if origin_img_np is not None:
            cv2.circle(origin_img_np, tuple(left_point), radius=4, color=(0,255,0), thickness=-1)
            cv2.circle(origin_img_np, tuple(right_point), radius=4, color=(0,0,255), thickness=-1)
            overlay_output_path_file = os.path.join(overlay_output_path, sample_id[0] + "_overlay.png")
            cv2.imwrite(overlay_output_path_file, origin_img_np)

print("Test inference completed. Coordinates and images saved.")

