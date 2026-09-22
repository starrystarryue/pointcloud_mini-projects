# README

打包提交包含 PointNet 和 PointNet++ SSG 在 ModelNet40 分类任务上的复现代码、评估脚本和部分评估结果。代码默认使用已经预采样好的 `modelnet40_normal_resampled` 数据集，可以在报告看到下载链接；模型权重也可在报告中的链接打开，解压缩后将 `checkpoints`、`modelnet40_normal_resampled `文件夹放到根目录下即可

- 运行命令均假设当前工作目录为本README文件所在目录（根目录）

## 1. 文件结构

```text
.
├── PointNet/
│   ├── dataset.py
│   ├── dataset_new.py
│   ├── evaluate.py
│   ├── model.py
│   └── train.py
├── PointNet++/
│   ├── dataset_ssg.py
│   ├── evaluate_ssg.py
│   └── train_ssg.py
├── eval_results/
│   └── pointnet2_ssg_best/
│       ├── metrics.json
│       └── predictions.csv
│   └── pointnet_resampled_best/
│       ├── metrics.json
│       └── predictions.csv
├── Homework3 报告.pdf
└── README.md
```

PointNet和PointNet++文件夹下为所有数据加载、训练、测试的脚本

ecal_results文件夹下为调用训练好的权重模型测试得到的结果，内部 pointnet2_ssg_best 和 pointnet_resampled_bes 子文件夹分别存放 PointNet++ 和 PointNet 的结果

数据集需要放在：

```text
./modelnet40_normal_resampled/
```

该数据集的文件结构形如：

```text
modelnet40_shape_names.txt
modelnet40_train.txt
modelnet40_test.txt
airplane/
bathtub/
...
```

## 2. checkpoints 文件夹结构

训练完成后，模型权重和训练日志建议保存为如下结构：

```text
./checkpoints/
├── pointnet_resampled/
│   ├── best_model.pth
│   ├── config.json
│   ├── history.csv
│   └── tensorboard/
└── pointnet2_ssg/
    ├── best_model.pth
    ├── config.json
    ├── history.csv
    └── tensorboard/
```

其中：

- `best_model.pth`：测试集准确率最高的模型权重
- `config.json`：训练时使用的参数配置
- `history.csv`：包含训练时每个 epoch 的训练和测试记录
- `tensorboard/`：TensorBoard 日志

如果权重以压缩包形式提交，例如 `checkpoints.zip`，请先解压到当前目录，使其路径为 `./checkpoints/...`。

## 3. PointNet 运行方式

### 训练 PointNet

在根目录中运行：

```bash
python PointNet/train.py --data_path ./modelnet40_normal_resampled --dataset normal_resampled --save_dir ./checkpoints/pointnet_resampled --epochs 250 --batch_size 32 --num_points 1024 --lr 0.001 --weight_decay 1e-4 --reg_weight 0.001 --step_size 20 --gamma 0.7 --dropout 0.3 --num_workers 4 --save_interval 50 --device auto
```

### 测试 PointNet

在根目录下，运行 `PointNet/evaluate.py` 直接调用训练好的权重模型测试：
```bash
python PointNet/evaluate.py --checkpoint ./checkpoints/pointnet_resampled/best_model.pth --data_path ./modelnet40_normal_resampled --dataset normal_resampled --split test --num_points 1024 --batch_size 32 --num_workers 4 --device auto --out_dir ./eval_results/pointnet_resampled_best
```

测试结果会输出整体准确率和平均类别准确率，并保存到：

```text
./eval_results/pointnet_resampled_best/
```

## 4. PointNet++ SSG 运行方式

### 训练 PointNet++ SSG

在根目录下运行：

```bash
python PointNet++/train_ssg.py --data_path ./modelnet40_normal_resampled --save_dir ./checkpoints/pointnet2_ssg --epochs 250 --batch_size 16 --num_points 1024 --lr 0.001 --weight_decay 0.0 --decay_step 200000 --decay_rate 0.7 --min_lr 1e-5 --bn_init_momentum 0.5 --bn_decay_rate 0.5 --bn_min_momentum 0.01 --dropout 0.5 --num_workers 4 --save_interval 50 --device auto
```

### 测试 PointNet++ SSG

根目录下，运行 `PointNet++/evaluate_ssg.py` 直接调用训练好的权重模型测试：
```bash
python PointNet++/evaluate_ssg.py --checkpoint ./checkpoints/pointnet2_ssg/best_model.pth --data_path ./modelnet40_normal_resampled --split test --num_points 1024 --batch_size 16 --num_workers 4 --device auto --out_dir ./eval_results/pointnet2_ssg_best
```

测试结果会输出 loss、整体准确率和平均类别准确率，并保存到：

```text
./eval_results/pointnet2_ssg_best/
```
