# Kaggle T4×2 双卡全自动挂机指南

在 Kaggle 上跑深度学习，核心逻辑是：**在网页端写好脚本 -> 丢给后台服务器跑（Commit） -> 跑完后提取 Output 续训。**

## 第一阶段：环境与数据准备

1. **新建 Notebook**：在 Kaggle 首页点击 `+ Create` -> `New Notebook`。
2. **开启双卡配置**：在右侧面板的 `Session Options` 中，找到 `Accelerator`，选择 **`GPU T4 x2`**。并开启 `Internet` 开关。
3. **挂载 ImageNet-100 数据集**：
   - 在右侧面板找到 `Input` 区域，点击 **`+ Add Input`**。
   - 搜索 `imagenet100`，找一个体积大约 13GB 左右的，点击 `Add`。
   - 记住挂载路径（通常是 `/kaggle/input/imagenet100/train` 等）。
4. **（可选）挂载 Colab 进度**：
   如果你想接着之前的进度跑，把之前下载的 `checkpoint.pth` 作为私有 Dataset 上传到 Kaggle，然后同样 `+ Add Input` 挂载进来。

---

## 第二阶段：Notebook 代码编写
在你的 Kaggle Notebook 中，按顺序创建以下 **3 个代码块（Cell）**：

### Cell 1: 下载代码
```python
!rm -rf /kaggle/working/dino
!git clone https://github.com/xbz123/dino-dense-degradation.git /kaggle/working/dino
```

### Cell 2: 准备断点续训
```python
import os
import shutil

output_dir = '/kaggle/working/dino_output'
os.makedirs(output_dir, exist_ok=True)

# 寻找上一次的存档点。如果你是第一次跑，这里什么都不会发生。
# 请将下面的路径替换为你实际挂载的历史 checkpoint 路径
old_ckpt_path = '/kaggle/input/YOUR_DATASET_NAME/checkpoint.pth' 

if os.path.exists(old_ckpt_path):
    print("找到历史进度，正在复制...")
    shutil.copy(old_ckpt_path, os.path.join(output_dir, 'checkpoint.pth'))
else:
    print("未找到历史进度，将从 Epoch 0 全新开始。")
```

### Cell 3: 启动双卡分布式训练
```python
%cd /kaggle/working/dino

# 注意：需将下面 data_path 替换为你第一阶段实际挂载的路径
!torchrun --nproc_per_node=2 main_dino.py \
    --arch vit_small \
    --patch_size 16 \
    --epochs 300 \
    --batch_size_per_gpu 64 \
    --accum_steps 2 \
    --warmup_teacher_temp_epochs 30 \
    --data_path /kaggle/input/imagenet100/train \
    --val_data_path /kaggle/input/imagenet100/val \
    --output_dir /kaggle/working/dino_output \
    --saveckp_freq 20 \
    --keep_last_ckpts 3 \
    --diag_every 5 \
    --attn_viz_every 25 \
    --use_fp16 true \
    --local_crops_number 4 \
    --num_workers 2 \
    --teacher_temp 0.07 
```

---

## 第三阶段：一键后台挂机 (Commit)

代码写完后，**千万不要在网页里一直开着跑**：

1. 点击 Kaggle 页面右上角的 **`Save Version`**。
2. 选择 **`Save & Run All (Commit)`**，点击 `Save`。
3. **关闭电脑**。Kaggle 会在后台连续跑 12 个小时，跑到断线后会自动封存你的 `dino_output` 文件夹作为这篇 Notebook 的 Output。

---

## 第四阶段：收获结果与开启下一轮循环

12 个小时后（或者第二天）：
1. 找到你昨天保存的 Notebook，在 **`Output`** 标签页里可以看到最新的 `checkpoint.pth`。
2. **开启下一轮**：点击 `Edit` 编辑该 Notebook。
3. 点击右侧 `+ Add Input` -> 选择 `Your Work` -> 挂载你**自己昨天生成的这个 Notebook**。
4. 把 Cell 2 里的 `old_ckpt_path` 改成这个新挂载进来的路径。
5. 再次点击 `Save Version` -> `Save & Run All`，开始新的 12 小时循环！
