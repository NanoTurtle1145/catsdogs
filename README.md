# catsdogs
基于Pytorch的猫狗图片识别程序
# 0x01 如何使用？
## 0x01-1 Linux
本程序对Linux比较友好，毕竟是用Linux写的。

### 0x01-1-1 训练模型（不推荐）
建议您创建虚拟环境。

首先，在终端输入

``` bash
pip install -r requirements.txt
```

注意，这个依赖列表是基我的机器的配置制作的，比如NVIDIA-CUDA之类的库。所以安装库时请确认自己的硬件。

待安装好依赖后，在终端输入

``` bash
python train.py
```

即可训练模型。

### 0x01-1-2 使用程序
本项目在上传代码时已经配好一个模型可供直接使用。文件名为``` best_catdog.pth ``` 。

但是我还没有针对此操作系统环境编写requirements.txt,所以请您自行安装依赖（比如先运行程序，如果报错，就安装相应库）。

该程序名为``` predict_gui.py ```。

## 0x01-2 Windows

### 0x01-2-1 训练模型（不推荐）
（同Linux）

首先，在终端输入

``` batch
pip install -r requirements.txt
```

注意，这个依赖列表是基我的机器的配置制作的，比如NVIDIA-CUDA之类的库。所以安装库时请确认自己的硬件。

待安装好依赖后，在终端输入

``` batch
python train.py
```

即可训练模型。

### 0x01-2-2 使用程序
本项目在上传代码时已经配好一个模型可供直接使用。文件名为``` best_catdog.pth ``` 。

我针对Windows编写了适用于Windows的requirements，名为``` requirements_win.txt ```

在终端输入

``` batch
pip install -r requirements_win.txt
```

即可，然后运行``` predict_gui.py ``` 即可。

该程序名为``` predict_gui.py ```。
