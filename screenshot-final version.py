import mss
import cv2
import numpy as np
import time
import keyboard
import os
from datetime import datetime


class ScreenCapture:
    def __init__(self, region=None, output_path="screenshots/"):
        """初始化屏幕截图工具"""
        self.region = region or self.get_fullscreen_region()
        self.output_path = output_path
        self.frame = None
        self.last_time = time.time()
        self.auto_interval = 0.05  # 自动截图间隔（0.05秒）
        self.auto_screenshots = []  # 自动截图缓存（最多保存4张）
        self.max_cache_size = 4

        # 定义保存目录
        self.directories = {
            'up': os.path.join(output_path, 'up'),
            'down': os.path.join(output_path, 'down'),
            'left': os.path.join(output_path, 'left'),
            'right': os.path.join(output_path, 'right'),
            'auto': os.path.join(output_path, 'auto')
        }

        # 创建所有目录
        for dir_path in self.directories.values():
            os.makedirs(dir_path, exist_ok=True)

        # 按键映射
        self.key_mapping = {'w': 'up', 's': 'down', 'a': 'left', 'd': 'right'}
        self.key_states = {key: False for key in self.key_mapping}

        # 自动截图计时
        self.last_auto_save = time.time()
        self.last_key_press_time = time.time()

        # 方向镜像映射
        self.mirror_mapping = {
            'up': 'up',
            'down': 'down',
            'left': 'right',
            'right': 'left',
            'auto': 'auto'
        }

    def get_fullscreen_region(self):
        """自动获取全屏区域"""
        with mss.mss() as sct:
            return sct.monitors[0]  # 全屏幕区域

    def capture_frame(self):
        """捕获当前屏幕帧并转换为灰度图"""
        with mss.mss() as sct:
            img = sct.grab(self.region)
            self.frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2GRAY)
            return self.frame

    def cache_auto_screenshot(self):
        """仅缓存自动截图，不保存到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.auto_screenshots.append({
            'frame': self.frame.copy(),
            'timestamp': timestamp
        })
        # 保持缓存大小
        if len(self.auto_screenshots) > self.max_cache_size:
            self.auto_screenshots.pop(0)
        return timestamp

    def create_composite_image(self, direction):
        """创建合成图（仅保存原始张量NPY）"""
        if len(self.auto_screenshots) < 4:
            print("缓存不足，至少需要4张自动截图")
            return None, None, None, None, None, None

        # 提取缓存中的截图
        img1 = self.auto_screenshots[-4]['frame']
        img2 = self.auto_screenshots[-2]['frame']
        img3 = self.frame

        h, w = img3.shape[:2]
        img1 = cv2.resize(img1, (w, h))
        img2 = cv2.resize(img2, (w, h))

        # 生成原始张量 [3, H, W]
        composite_tensor = np.stack([img1, img2, img3], axis=0)

        # 获取镜像方向
        mirror_direction = self.mirror_mapping[direction]

        # 生成镜像张量
        mirrored_tensor = np.flip(composite_tensor, axis=2).copy()

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # 保存原始张量
        dir_path = self.directories[direction]
        tensor_filename = os.path.join(dir_path, f"tensor_{direction}_{timestamp}.npy")
        np.save(tensor_filename, composite_tensor)

        # 保存镜像张量
        mirror_dir_path = self.directories[mirror_direction]
        mirror_tensor_filename = os.path.join(mirror_dir_path, f"tensor_{mirror_direction}_{timestamp}_mirror.npy")
        np.save(mirror_tensor_filename, mirrored_tensor)

        print(f"原始张量保存: {tensor_filename}")
        print(f"镜像张量保存: {mirror_tensor_filename}")

        return composite_tensor, None, tensor_filename, mirrored_tensor, None, mirror_tensor_filename

    def run(self):
        """启动程序主循环（无窗口显示）"""
        print("程序启动 - 功能说明:")
        print("  自动模式: 每0.05秒缓存截图（内存中，最多4张）")
        print("  按键模式: W/S/A/D 生成合成图和镜像图")
        print("  无操作模式: 1秒无按键自动生成合成图和镜像图")
        print("按 ESC 退出程序")

        while not keyboard.is_pressed('esc'):
            self.capture_frame()

            # 处理自动缓存
            if time.time() - self.last_auto_save >= self.auto_interval:
                self.cache_auto_screenshot()
                self.last_auto_save = time.time()

            # 检测按键操作
            key_pressed = False
            for key, direction in self.key_mapping.items():
                if keyboard.is_pressed(key) and not self.key_states[key]:
                    self.key_states[key] = True
                    self.last_key_press_time = time.time()
                    key_pressed = True
                    # 生成合成图和镜像图
                    self.create_composite_image(direction)

            # 重置按键状态
            for key in self.key_mapping:
                if not keyboard.is_pressed(key):
                    self.key_states[key] = False

            # 检测无按键操作时间
            idle_time = time.time() - self.last_key_press_time
            if idle_time >= 1.0 and len(self.auto_screenshots) >= 4:
                self.last_key_press_time = time.time()
                print("检测到1秒无按键操作，自动生成合成图")
                self.create_composite_image('auto')

            time.sleep(0.01)  # 降低CPU占用

        print("程序退出")

    @staticmethod
    def load_composite_tensor(file_path):
        """加载原始张量"""
        try:
            tensor = np.load(file_path)
            print(f"成功加载张量，维度: {tensor.shape}")
            return tensor
        except Exception as e:
            print(f"加载失败: {e}")
            return None


# ====================== 使用示例 ======================
if __name__ == "__main__":
    # 自定义截图区域（示例：全屏）
    custom_region = {
        "top": 0,
        "left": 0,
        "width": 3840,  # 屏幕宽度
        "height": 2160  # 屏幕高度
    }

    # 创建实例
    sc = ScreenCapture(
        region=custom_region,
        output_path=r"C:\Users\4h55\Pictures\ScreenCaptures"
    )

    # 启动程序（无窗口显示）
    sc.run()