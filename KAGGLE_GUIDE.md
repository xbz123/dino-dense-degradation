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

### Cell 2：确认当前代码版本

```python
%cd /kaggle/working/dino
!git rev-parse HEAD
!python -m py_compile main_dino.py utils.py dense_diagnostics.py
```

当前仓库已经在 `utils.py` 和评估脚本中使用 `torch.load(..., weights_only=False)`，不需要再在 Kaggle Notebook 里手动修改源码。

### Cell 3：设置断点续训路径

```python
import os

output_dir = '/kaggle/working/dino_output'
os.makedirs(output_dir, exist_ok=True)

# 如果你挂载了上一轮 Notebook Output 或 checkpoint Dataset，请把这里改成实际路径。
# 示例：
#   '/kaggle/input/dino-train-v2/dino_output/checkpoint.pth'
#   '/kaggle/input/dino-ckp/checkpoint0180.pth'
# 第一次从头训练时保留为空字符串。
resume_ckpt = ''

if resume_ckpt and os.path.exists(resume_ckpt):
    resume_from_arg = f'--resume_from {resume_ckpt}'
    print('Resume checkpoint:', resume_ckpt)
else:
    resume_from_arg = ''
    print('No resume checkpoint configured; training will start from output_dir/checkpoint.pth if it exists, otherwise epoch 0.')
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
    {resume_from_arg} \
    --saveckp_freq 10 \
    --keep_last_ckpts 0 \
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
- `--resume_from`：直接读取挂载的历史 checkpoint；不需要复制覆盖 `/kaggle/working/dino_output/checkpoint.pth`
- `--saveckp_freq 10`：每 10 个 epoch 保存一次历史 checkpoint
- `--keep_last_ckpts 0`：保留所有历史 checkpoint，便于后续 dense degradation sweep
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

1. 打开你的 Notebook，在 **`Output`** 标签页找到 `dino_output/checkpoint.pth` 或最新的 `dino_output/checkpointXXXX.pth`
2. **开启下一轮**：
   - 点击 `Edit` 回到编辑界面
   - 点击右侧 `+ Add Input` → 选择 `Your Work` → 挂载你**自己上一轮的 Notebook**
   - 把 Cell 3 里的 `resume_ckpt` 改成新挂载的路径（通常是 `/kaggle/input/你的notebook名/dino_output/checkpoint.pth`，也可以选择最新的 `checkpointXXXX.pth`）
   - 再次 `Save Version` → `Save & Run All`，开始新的 12 小时挂机

---

## 第五阶段：COCO-Stuff selected-checkpoint 验证

这个阶段用于验证 PASCAL VOC 上看到的 post-epoch-180 dense drop 是否也出现在 COCO-Stuff 上。它**不重新训练 DINO backbone**，只对已保存的 checkpoint 做 frozen-backbone linear probing。

### 需要提前准备的 Kaggle Input

1. **代码仓库**
   - 使用 Cell 1 中 clone 的 GitHub 仓库即可。
   - 需要包含：
     ```text
     eval_coco_stuff_dense.py
     eval_voc_dense.py
     dense_eval_utils.py
     ```

2. **DINO checkpoints**
   - 挂载包含历史 checkpoint 的 Kaggle Dataset。
   - 目录里至少要有：
     ```text
     checkpoint0050.pth
     checkpoint0080.pth
     checkpoint0180.pth
     checkpoint0220.pth
     checkpoint0300.pth
     checkpoint0318.pth
     ```
   - 示例路径：
     ```text
     /kaggle/input/datasets/bingzhouxie/dinockp/
     ```

3. **COCO-Stuff 数据**
   - 挂载 COCO-Stuff train/val image 和 annotation mask。
   - 推荐在 Drive/Colab 中整理为同样结构后上传到 Kaggle Dataset：
     ```text
     /content/drive/MyDrive/coco_stuff/
       images/train2017/*.jpg
       images/val2017/*.jpg
       annotations/train2017/*.png
       annotations/val2017/*.png
     ```
   - 在 Kaggle 中示例路径可能是：
     ```text
     /kaggle/input/coco-stuff/coco_stuff/
     ```
   - 如果使用 Kaggle 上的 `dntai2/cocostuff-10k-v1-1`，当前 evaluator 也支持它的 10K v1.1 结构：
     ```text
     /kaggle/input/datasets/dntai2/cocostuff-10k-v1-1/
       cocostuff-10k-v1.1/images/*.jpg
       cocostuff-10k-v1.1/annotations/*.png
       cocostuff-10k-v1.1/imageLists/*.txt
     ```

4. **已有 VOC/DSE 结果（可选但推荐）**
   - 如果要生成 `coco_voc_dse_comparison.csv`，挂载已有 raw/L2 run 输出：
     ```text
     voc_all_checkpoints/voc_miou_results.json
     figures/combined_dense_summary.csv
     ```

### Smoke test：先只跑 180 和 318

先确认数据路径、mask label、nearest-neighbor resize、linear head 训练和 mIoU 计算都正常：

```python
%cd /kaggle/working/dino

!python eval_coco_stuff_dense.py \
    --ckpt_dir /kaggle/input/datasets/bingzhouxie/dinockp \
    --coco_root /kaggle/input/datasets/dntai2/cocostuff-10k-v1-1 \
    --epochs 180,318 \
    --img_size 336 \
    --batch_size 64 \
    --train_epochs 1 \
    --lr 0.0025 \
    --feature_dtype float16 \
    --max_train_images 2000 \
    --max_val_images 500 \
    --output_dir /kaggle/working/dino_dense_degradation_eval/to_epoch_0318_raw_l2/coco_stuff_selected_smoke
```

Smoke test 通过后，检查输出：

```text
coco_stuff_miou_results.json
dense_degradation_coco_stuff.png
coco_voc_dse_comparison.csv
coco_stuff_summary.md
```

### 正式 selected-checkpoint run

正式最小验证跑 6 个 checkpoint：

```python
%cd /kaggle/working/dino

!python eval_coco_stuff_dense.py \
    --ckpt_dir /kaggle/input/datasets/bingzhouxie/dinockp \
    --coco_root /kaggle/input/datasets/dntai2/cocostuff-10k-v1-1 \
    --epochs 50,80,180,220,300,318 \
    --img_size 336 \
    --batch_size 64 \
    --train_epochs 15 \
    --lr 0.0025 \
    --feature_dtype float16 \
    --voc_results_json /kaggle/input/dino-eval-results/to_epoch_0318_raw_l2/voc_all_checkpoints/voc_miou_results.json \
    --dense_summary_csv /kaggle/input/dino-eval-results/to_epoch_0318_raw_l2/figures/combined_dense_summary.csv \
    --output_dir /kaggle/working/dino_dense_degradation_eval/to_epoch_0318_raw_l2/coco_stuff_selected
```

判断标准：

- 如果 COCO-Stuff 也在 epoch 180 附近最好，并在 epoch 318 下降，说明 VOC drop 不是单一数据集现象。
- 如果 COCO-Stuff 不下降，当前结论要保持谨慎，VOC drop 可能和 dataset/probe sensitivity 有关。
- 如果曲线很 noisy，再只对 `180 / 300 / 318` 跑 3 个 seed，而不是立即扩大到全 checkpoint sweep。

---

## 性能参考

| 平台 | GPU | 步数/epoch | 每 epoch 耗时 |
|------|-----|-----------|-------------|
| Colab Free | T4 ×1 | 1979 | ~66 分钟 |
| **Kaggle Free** | **T4 ×2** | **989** | **~40 分钟** |

- 300 epochs 总计约 **200 小时**
- Kaggle 每周 30 小时 GPU 额度 → 约 **7 周**完成
- 可与 Colab 交替使用，checkpoint 完全兼容互通
