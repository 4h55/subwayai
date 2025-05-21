import mss
import cv2
import numpy as np
import os
import time
import keyboard
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from pynput.keyboard import Controller
from threading import Thread
from queue import Queue  # 替换为从queue模块导入
from datetime import datetime


# ====================== 模型定义 ======================
class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super(SEBlock, self).__init__()
        mid_channels = int(in_channels * se_ratio)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.Hardswish(),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        se = self.squeeze(x).view(b, c, 1, 1)
        se = self.excitation(se)
        return x * se.expand_as(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, se_ratio=0.25):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        expand_channels = in_channels * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, expand_channels, 1),
            nn.BatchNorm2d(expand_channels),
            nn.Hardswish(),
            nn.Conv2d(expand_channels, expand_channels, 3, stride=stride, padding=1, groups=expand_channels),
            nn.BatchNorm2d(expand_channels),
            nn.Hardswish(),
            SEBlock(expand_channels, se_ratio=se_ratio),
            nn.Conv2d(expand_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return x + self.conv(x) if self.use_residual else self.conv(x)


class TimeFocusedModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.time_importance = nn.Parameter(torch.tensor([0.5, 0.3, 3.0], dtype=torch.float32))

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
            InvertedResidual(32, 32, 2, 4),
            InvertedResidual(32, 32, 1, 4),
            InvertedResidual(32, 64, 2, 4),
            InvertedResidual(64, 64, 1, 3),
            InvertedResidual(64, 128, 2, 3),
            InvertedResidual(128, 128, 1, 2),
            InvertedResidual(128, 256, 2, 4),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(256, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.Hardswish(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.time_raw_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
            InvertedResidual(16, 24, 2, 3),
            InvertedResidual(24, 24, 1, 3),
            InvertedResidual(24, 32, 2, 3),
            InvertedResidual(32, 32, 1, 3),
            InvertedResidual(32, 64, 2, 3),
            nn.AdaptiveAvgPool2d(1)
        )

        self.time_conv = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.Hardswish(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024 + 128, 512),
            nn.BatchNorm1d(512),
            nn.Hardswish(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Hardswish(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        spatial_feat = self.spatial_encoder(x[:, 2:3, :, :]).flatten(1) * self.time_importance[2]
        time_feats = []
        for t in range(3):
            feat = self.time_raw_encoder(x[:, t:t + 1, :, :]).flatten(1) * self.time_importance[t]
            time_feats.append(feat)
        time_seq = torch.stack(time_feats, dim=2)
        time_feat = self.time_conv(time_seq).squeeze(2)
        return self.classifier(torch.cat([spatial_feat, time_feat], dim=1))


# ====================== 游戏控制类 ======================
class GameController:
    def __init__(self):
        self.keyboard = Controller()
        self.action_map = {0: None, 1: 's', 2: 'a', 3: 'd', 4: 'w'}

    def perform_action(self, action):
        key = self.action_map.get(action, None)
        if key:
            self.keyboard.press(key)
            time.sleep(0.05)  # 按键持续时间
            self.keyboard.release(key)
            print(f"执行键盘动作: {key}")


# ====================== 截图与推理整合 ======================
class RealTimeController:
    def __init__(self, output_path="sc/"):
        self.output_path = output_path
        self.target_size = (128, 128)
        self.auto_interval = 0.05
        self.max_cache = 4
        self.auto_screenshots = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(r"C:\Users\bxd\Desktop\CurrentBestModel.pth")
        self.controller = GameController()  # 使用自定义的GameController
        self.file_queue = Queue()  # 线程间通信队列
        self._init_dirs()

    def _init_dirs(self):
        self.directories = {'auto': os.path.join(self.output_path, '0')}
        for dir_path in self.directories.values():
            os.makedirs(dir_path, exist_ok=True)

    def load_model(self, model_path):
        model = TimeFocusedModel(num_classes=5)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def capture_loop(self):
        """截图线程主函数"""
        sc = mss.mss()
        region = {"top": 0, "left": 128, "width": 896, "height": 768}
        last_save = time.time()

        while not keyboard.is_pressed('esc'):
            # 捕获屏幕并转换为灰度图
            img = np.array(sc.grab(region))
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

            # 维护截图缓存
            self.auto_screenshots.append(gray)
            if len(self.auto_screenshots) > self.max_cache:
                # 生成合成图并保存NPY
                composite = np.stack([
                    self.auto_screenshots[0],  # t-3帧
                    self.auto_screenshots[2],  # t-1帧
                    self.auto_screenshots[-1]  # 当前帧
                ], axis=0)
                resized = self.resize_tensor(composite)
                filename = self.save_npy(resized)
                self.file_queue.put(filename)  # 将文件路径放入队列
                self.auto_screenshots.pop(0)

            # 控制截图频率
            time.sleep(max(0, self.auto_interval - (time.time() - last_save)))
            last_save = time.time()

    def resize_tensor(self, tensor):
        """调整尺寸"""
        return np.array([cv2.resize(frame, self.target_size) for frame in tensor])

    def save_npy(self, tensor):
        """保存NPY文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(self.directories['auto'], f"tensor_{timestamp}_128x128.npy")
        np.save(filename, tensor)
        print(f"保存文件: {filename}")
        return filename

    def inference_loop(self):
        """推理线程主函数"""
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        while not keyboard.is_pressed('o'):
            if not self.file_queue.empty():
                file_path = self.file_queue.get()
                try:
                    # 预处理并推理
                    inputs = self.process_npy(file_path, transform)
                    with torch.no_grad():
                        output = self.model(inputs)
                        action = torch.argmax(output, dim=1).item()
                    self.controller.perform_action(action)  # 调用自定义的perform_action
                    print(f"执行动作: {action}")
                except Exception as e:
                    print(f"推理失败: {e}")

    def process_npy(self, npy_path, transform):
        depth = np.load(npy_path).astype(np.float32)
        tensor = torch.from_numpy(depth).unsqueeze(0).to(self.device)  # (1, 3, 128, 128)
        return transform(tensor)

    def run(self):
        """启动双线程"""
        print("系统启动，按ESC退出")
        capture_thread = Thread(target=self.capture_loop, daemon=True)
        inference_thread = Thread(target=self.inference_loop, daemon=True)

        capture_thread.start()
        inference_thread.start()

        while not keyboard.is_pressed('esc'):
            time.sleep(0.1)
        print("系统退出")


# ====================== 主程序入口 ======================
if __name__ == "__main__":
    rt_controller = RealTimeController(output_path=r"C:\Users\bxd\Desktop\sc")
    rt_controller.run()
#dddddd