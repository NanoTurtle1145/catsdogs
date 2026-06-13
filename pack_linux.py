#!/usr/bin/env python3
"""
PyInstaller 打包脚本 (onedir + windowed)
- 入口文件: predict_gui.py
- 输出文件夹: dist/CatDogClassifier/
- 自动排除训练/GPU/开发相关模块，减小体积
"""

import os
import sys
import subprocess

# ======================== 配置项 ========================
ENTRY_FILE = "predict_gui.py"      # 你的主程序入口文件
OUTPUT_NAME = "CatDogClassifier"   # 输出可执行文件名称
EXTRA_ARGS = []                    # 可添加额外参数，如 --icon=icon.ico
# ========================================================

# 需要排除的顶级模块和子模块列表（训练/GPU/开发相关）
EXCLUDE_MODULES = [
    # CUDA / GPU 训练库
    "cuda", "nvidia", "triton", "nvrtc", "cudart", "cublas", "cudnn", "nccl",
    # PyTorch 训练子模块（保留核心推理）
    "torch.distributed", "torch.jit", "torch.quantization",
    "torch.onnx", "torch.optim", "torch.sparse", "torch.testing",
    "torch._dynamo", "torch._inductor", "torch._lazy", "torch._functorch",
    "torch._export", "torch._higher_order_ops", "torch._prims", "torch._refs",
    "torch.compiler", "torch.distributions", "torch.backends.cudnn",
    # 开发工具 / 文档 / 测试
    "setuptools", "pip", "pytest", "IPython", "jupyter", "notebook",
    "tensorboard", "wheel", "docutils", "sphinx",
    "matplotlib.mpl-data.sample_data",   # matplotlib 示例数据
]

def main():
    if not os.path.isfile(ENTRY_FILE):
        print(f"错误: 找不到入口文件 {ENTRY_FILE}")
        sys.exit(1)

    # 构建 pyinstaller 命令
    cmd = [
        "pyinstaller",
        "--onedir",           # -D
        "--windowed",        # -w
        "--name", OUTPUT_NAME,
        "--clean",
        "--noconfirm",
    ]

    # 添加排除模块参数
    for mod in EXCLUDE_MODULES:
        cmd.extend(["--exclude-module", mod])

    # 添加额外参数
    cmd.extend(EXTRA_ARGS)

    # 入口文件放在最后
    cmd.append(ENTRY_FILE)

    print("正在执行打包命令:")
    print(" ".join(cmd))
    print("\n请稍候，打包过程可能需要几分钟...\n")

    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"\n✅ 打包成功！输出目录: ./dist/{OUTPUT_NAME}/")
        print(f"可执行文件: ./dist/{OUTPUT_NAME}/{OUTPUT_NAME}")
    else:
        print("\n❌ 打包失败，请检查上述错误信息。")

if __name__ == "__main__":
    main()
