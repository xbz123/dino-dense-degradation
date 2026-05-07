# Kaggle T4×2 双卡训练指南

在 Kaggle 上跑深度学习，核心逻辑是：**在网页端写好脚本 → 丢给后台服务器跑（Commit） → 跑完后提取 Output 续训。**

---

## 第一阶段：环境与数据准备

1. **新建 Notebook**：在 Kaggle 首页点击 `+ Create` → `New Notebook`。
2. **开启双卡配置**：在右侧面板底部的 `Session Options` 中：
   - `Accelerator` 选择 **`GPU T4 x2`**
   - `Internet` 开关**打开**（用于 clone GitHub 代码）
3. **挂载 ImageNet-100 数据集**：
   - 在右侧面板找到 `Input` 区域，点击 **`+ Add Input`**
   - 搜索 **`wilyzh/imagenet100`**，点击 `Add`
   - 挂载后的数据路径为：
     ```
     训练集: /kaggle/input/datasets/wilyzh/imagenet100/ImageNet100/train
     验证集: /kaggle/input/datasets/wilyzh/imagenet100/ImageNet100/val
     ```
4. **（可选）挂载历史 checkpoint**：
   - 如果你想接着之前的进度跑（从 Colab 或上一轮 Kaggle），把 `checkpoint.pth` 作为私有 Dataset 上传到 Kaggle
   - 然后同样通过 `+ Add Input` 挂载进来

---

## 第二阶段：Notebook 代码编写

在 Kaggle Notebook 中，按顺序创建以下 **5 个代码块（Cell）**：

### Cell 1：下载代码

```python
%cd /kaggle/working
!rm -rf /kaggle/working/dino
!git clone https://github.com/xbz123/dino-dense-degradation.git /kaggle/working/dino
```

> ⚠️ 必须先 `%cd /kaggle/working`，否则删除 dino 文件夹后 shell 会找不到当前目录而报错。

### Cell 2：修复 PyTorch 2.6 兼容性

Kaggle 使用 PyTorch 2.6+，其 `torch.load` 默认开启了 `weights_only=True`，会导致加载 checkpoint 时报错。运行以下代码自动修复：

```python
path = '/kaggle/working/dino/utils.py'
with open(path, 'r') as f:
    code = f.read()
code = code.replace(
    'torch.load(ckp_path, map_location="cpu")',
    'torch.load(ckp_path, map_location="cpu", weights_only=False)'
)
code = code.replace(
    'torch.load(pretrained_weights, map_location="cpu")',
    'torch.load(pretrained_weights, map_location="cpu", weights_only=False)'
)
with open(path, 'w') as f:
    f.write(code)
print("utils.py 已修复！")
```

### Cell 3：准备断点续训

```python
import os
import shutil

output_dir = '/kaggle/working/dino_output'
os.makedirs(output_dir, exist_ok=True)

# 如果你挂载了历史 checkpoint，请将下面的路径替换为实际的挂载路径
# 例如: '/kaggle/input/dino-kpt/checkpoint.pth'
# 如果是第一次从头开始跑，这段代码会自动跳过
old_ckpt_path = '/kaggle/input/dino_kpt/checkpoint.pth'

if os.path.exists(old_ckpt_path):
    print("找到历史进度，正在复制...")
    shutil.copy(old_ckpt_path, os.path.join(output_dir, 'checkpoint.pth'))
    print("复制完成！")
else:
    print("未找到历史进度，将从 Epoch 0 全新开始。")
```

### Cell 4：确认数据路径（可选，首次运行建议执行）

```python
import os
train_path = '/kaggle/input/datasets/wilyzh/imagenet100/ImageNet100/train'
val_path = '/kaggle/input/datasets/wilyzh/imagenet100/ImageNet100/val'
print(f"Train classes: {len(os.listdir(train_path))}")
print(f"Val classes: {len(os.listdir(val_path))}")
# 应该输出 100
```

### Cell 5：启动双卡分布式训练

```python
%cd /kaggle/working/dino

!torchrun --nproc_per_node=2 main_dino.py \
    --arch vit_small \
    --patch_size 16 \
    --epochs 300 \
    --batch_size_per_gpu 64 \
    --accum_steps 2 \
    --warmup_teacher_temp_epochs 30 \
    --data_path /kaggle/input/datasets/wilyzh/imagenet100/ImageNet100/train \
    --val_data_path /kaggle/input/datasets/wilyzh/imagenet100/ImageNet100/val \
    --output_dir /kaggle/working/dino_output \
    --saveckp_freq 20 \
    --keep_last_ckpts 3 \
    --diag_every 5 \
    --attn_viz_every 25 \
    --use_fp16 true \
    --local_crops_number 4 \
    --num_workers 2 \
    --teacher_temp 0.07 \
    --norm_last_layer false
```

**参数说明：**
- `--nproc_per_node=2`：使用 2 张 T4 GPU 并行训练
- `--batch_size_per_gpu 64 × 2卡 × --accum_steps 2 = 256` 等效批次大小
- `--local_crops_number 4`：减少到 4 个局部裁切（节省约 25% 计算）
- `--diag_every 5`：每 5 个 epoch 运行一次稠密退化诊断
- 显存占用约 9.6 GB / 16 GB，每 epoch 约 40 分钟

---

## 第三阶段：一键后台挂机（Commit）

代码写完后，**千万不要在网页里一直开着跑**（关网页后环境会被重置）：

1. 点击 Kaggle 页面右上角的 **`Save Version`**
2. 选择 **`Save & Run All (Commit)`**，点击 `Save`
3. **放心关闭电脑**。Kaggle 会在后台连续跑最多 12 个小时
4. 跑完（或超时中止）后，`/kaggle/working/` 下的所有文件会被永久封存为该 Notebook 的 Output

---

## 第四阶段：收获结果与开启下一轮循环

12 个小时后（或第二天）：

1. 打开你的 Notebook，在 **`Output`** 标签页找到 `dino_output/checkpoint.pth`
2. **开启下一轮**：
   - 点击 `Edit` 回到编辑界面
   - 点击右侧 `+ Add Input` → 选择 `Your Work` → 挂载你**自己上一轮的 Notebook**
   - 把 Cell 3 里的 `old_ckpt_path` 改成新挂载的路径（通常是 `/kaggle/input/你的notebook名/dino_output/checkpoint.pth`）
   - 再次 `Save Version` → `Save & Run All`，开始新的 12 小时挂机

---

## 性能参考

| 平台 | GPU | 步数/epoch | 每 epoch 耗时 |
|------|-----|-----------|-------------|
| Colab Free | T4 ×1 | 1979 | ~66 分钟 |
| **Kaggle Free** | **T4 ×2** | **989** | **~40 分钟** |

- 300 epochs 总计约 **200 小时**
- Kaggle 每周 30 小时 GPU 额度 → 约 **7 周**完成
- 可与 Colab 交替使用，checkpoint 完全兼容互通
