import os
import random
import argparse
from pathlib import Path


def delete_npy_files(directory, probability, dry_run=False):
    """
    按指定概率随机删除目录中的npy文件

    参数:
    directory (str): 要处理的目录路径
    probability (float): 删除文件的概率(0.0-1.0)
    dry_run (bool): 是否进行干跑(只显示操作不实际删除)
    """
    # 验证概率范围
    if not (0.0 <= probability <= 1.0):
        raise ValueError("概率必须在0.0到1.0之间")

    # 获取所有npy文件
    npy_files = list(Path(directory).glob("*.npy"))

    if not npy_files:
        print(f"在目录 {directory} 中未找到npy文件")
        return

    print(f"找到 {len(npy_files)} 个npy文件")

    deleted_count = 0

    # 遍历处理每个文件
    for file_path in npy_files:
        # 随机决定是否删除
        if random.random() < probability:
            if dry_run:
                print(f"[干跑] 会删除: {file_path}")
            else:
                try:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"删除文件 {file_path} 时出错: {e}")

    print(f"操作完成。共删除 {deleted_count} 个文件")


if __name__ == "__main__":
    # 设置默认参数值（可直接在PyCharm中运行）
    DEFAULT_DIRECTORY = r"C:\Users\xiang\OneDrive\桌面\subwayai\pythonProject\subwAI-surfer\data\4"  # 修改为你的目录路径
    DEFAULT_PROBABILITY = 0.1  # 修改为你需要的概率值
    DEFAULT_DRY_RUN = False  # 修改为False以执行实际删除

    parser = argparse.ArgumentParser(description="随机删除npy文件")
    parser.add_argument("-d", "--directory", default=DEFAULT_DIRECTORY,
                        help="要处理的目录路径")
    parser.add_argument("-p", "--probability", type=float, default=DEFAULT_PROBABILITY,
                        help="删除文件的概率(0.0-1.0)")
    parser.add_argument("--dry-run", action="store_true", default=DEFAULT_DRY_RUN,
                        help="进行干跑(只显示操作不实际删除)")

    args = parser.parse_args()

    try:
        delete_npy_files(args.directory, args.probability, args.dry_run)
    except Exception as e:
        print(f"程序执行出错: {e}")