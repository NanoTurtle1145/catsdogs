#!/bin/bash
# 在干净 CPU 环境下运行此脚本
# 打包输出文件夹: dist/CatDogClassifier/

pyinstaller \
    --onedir \
    --windowed \
    --name CatDogClassifier \
    --clean \
    --noconfirm \
    --exclude-module cuda \
    --exclude-module nvidia \
    --exclude-module triton \
    --exclude-module torch.distributed \
    --exclude-module torch.jit \
    --exclude-module torch.quantization \
    --exclude-module torch.onnx \
    --exclude-module torch.optim \
    --exclude-module torch.sparse \
    --exclude-module torch.testing \
    --exclude-module torch._dynamo \
    --exclude-module torch._inductor \
    --exclude-module torch._lazy \
    --exclude-module torch._functorch \
    --exclude-module torch._export \
    --exclude-module torch._higher_order_ops \
    --exclude-module torch._prims \
    --exclude-module torch._refs \
    --exclude-module torch.compiler \
    --exclude-module torch.distributions \
    --exclude-module torch.backends.cudnn \
    --exclude-module setuptools \
    --exclude-module pip \
    --exclude-module pytest \
    --exclude-module IPython \
    --exclude-module jupyter \
    --exclude-module tensorboard \
    --exclude-module matplotlib.mpl-data.sample_data \
    predict_gui.py

echo "打包完成，输出目录: dist/CatDogClassifier/"
