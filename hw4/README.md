# README

本文件说明提交文件夹中的代码结构、各文件作用以及运行方式，所有命令默认在本markdown文件所在目录下执行。

## 1. 环境依赖

实验环境为python 3.11，代码主要依赖以下 Python 库，可以创建一个conda虚拟环境：

```bash
conda create -n hw4 python=3.11
conda activate hw4
pip install torch numpy tqdm scikit-image
```

如果使用 GPU，请确保 PyTorch 已正确安装 CUDA 版本。程序会默认选择 `cuda`，若没有可用 GPU，则自动使用 `cpu`。

## 2. 文件结构

```text
handin/
+-- README.md
+-- Homework4 Report.pdf
+-- sdf_model.py
+-- train.py
+-- test.py
+-- getmesh.py
+-- data/	#【目前为空】
|   +-- <uid>/
|   |   +-- <uid>.obj
|   |   +-- pointcloud.npz
|   |   +-- sdf.npz
|   +-- ...
+-- outputs/
    +-- checkpoints/			#【目前为空】
    +-- checkpoints_Fourier/	#【目前为空】
    +-- logs/
    +-- logs_Fourier/
    +-- sdf/
    +-- sdf_Fourier/
    +-- mesh/
    +-- mesh_Fourier/
```

## 3. 各文件作用

`Homework4 Report.pdf`  ：作业报告

`sdf_model.py`  ：定义 SDF 隐式表示网络。包括基础 MLP、Fourier feature 编码模块、模型构建函数、checkpoint 加载函数、数据读取函数以及测试阶段生成规则网格点的函数。

`train.py`  ：训练脚本。对 `data/` 中每个 shape 单独训练一个 SDF MLP，并保存 checkpoint 和训练日志。训练损失包括 SDF loss 和 gradient loss。

`test.py`  ：测试与重建入口脚本。加载训练好的 checkpoint，在规则三维网格上预测 SDF 值，保存 `.npy` SDF grid 和 `.json` metadata，然后调用 `getmesh.py` 进行 marching cubes 重建。

`getmesh.py`  ：根据保存好的 SDF grid 提取零水平集表面。内部使用 `skimage.measure.marching_cubes`，并将结果写为 `.obj` mesh。

`data/`  ：训练和测试数据，即从教学网上下载的作业包时的无任何改动的data文件夹，这里是一个<u>**空文件夹**</u>

`outputs/checkpoints/`  ：基础 MLP 训练得到的模型权重，<u>**目前为空文件夹**</u>。可以在这个链接下载权重模型：

> https://disk.pku.edu.cn/link/AAFE0DE44E462F4CE392FA7B791A3F9A71
> 文件夹名：checkpoints
> 有效期限：永久有效

`outputs/checkpoints_Fourier/`  ：加入 Fourier feature 后训练得到的模型权重，<u>**目前为空文件夹**</u>。可以在这个链接下载权重模型：

> https://disk.pku.edu.cn/link/AAD3525F15A71540E4A4DF2E4800BB01BA
> 文件夹名：checkpoints_Fourier
> 有效期限：永久有效

`outputs/logs/`  ：基础 MLP 的训练日志，包括每个 shape 的 loss history、训练时间、最终 loss 等。

`outputs/logs_Fourier/`  ：Fourier feature 模型的训练日志。

`outputs/sdf/`  ：基础 MLP 在测试阶段预测得到的 SDF grid 及对应 metadata。

`outputs/sdf_Fourier/`  ：Fourier feature 模型预测得到的 SDF grid 及对应 metadata。

`outputs/mesh/`  ：基础 MLP 由 marching cubes 重建得到的 `.obj` mesh。

`outputs/mesh_Fourier/`  ：Fourier feature 模型由 marching cubes 重建得到的 `.obj` mesh。

## 4. 运行方式

### 4.1 训练基础 MLP

```bash
python train.py --fourier-features 0
```

默认设置为：

```text
hidden_dim = 512
layers = 6
steps = 20000
batch_size = 8192
lr = 1e-3
lambda_grad = 0.1
```

训练完成后，模型权重保存在：

```text
outputs/checkpoints/
```

训练日志保存在：

```text
outputs/logs/
```

### 4.2 训练 Fourier Feature 模型

```text
python train.py --checkpoint-dir outputs/checkpoints_Fourier --log-dir outputs/logs_Fourier --fourier-features 6 --fourier-scale 1.0
```

训练完成后，模型权重保存在：

```text
outputs/checkpoints_Fourier/
```

训练日志保存在：

```text
outputs/logs_Fourier/
```

### 4.3 使用基础 MLP 权重进行 SDF 预测和 mesh 重建

```bash
python test.py --resolution 256
```

该命令会加载 `outputs/checkpoints/` 中的基础 MLP 权重，并输出：

```text
outputs/sdf/
outputs/mesh/
```

其中 `outputs/sdf/` 保存预测得到的 SDF grid，`outputs/mesh/` 保存 marching cubes 重建得到的 `.obj` mesh。

### 4.4 使用 Fourier Feature 权重进行 SDF 预测和 mesh 重建

```text
python test.py --resolution 256 --checkpoint-dir outputs/checkpoints_Fourier --mesh-dir outputs/mesh_Fourier --sdf-dir outputs/sdf_Fourier --fourier-features 6 --fourier-scale 1.0
```

该命令会加载 Fourier feature 模型权重，并输出：

```text
outputs/sdf_Fourier/
outputs/mesh_Fourier/
```

### 4.5 仅预测 SDF，不重建 mesh

如果只希望保存 SDF grid，不调用 marching cubes生成.obj文件，可以运行：

```bash
python test.py --resolution 256 --no-mesh
```

### 4.6 单独从 SDF grid 重建 mesh

也可以直接调用 `getmesh.py`：

```text
python getmesh.py --sdf outputs/sdf/<uid>.npy --meta outputs/sdf/<uid>.json --out outputs/mesh/<uid>_predict.obj --iso-level 0.0
```

其中 `<uid>` 替换为具体 shape 的文件夹名。

## 5. 输出说明

运行 `test.py` 时，终端会输出每个 shape 的 SDF 预测时间和 marching cubes 重建时间，例如：

```text
[time] <uid> model SDF prediction: 1.178s
[time] <uid> marching cubes reconstruction: 0.381s
```

其中 `model SDF prediction` 表示模型在规则网格上预测 SDF 的耗时，`marching cubes reconstruction` 表示根据 SDF grid 提取 mesh 的耗时。
