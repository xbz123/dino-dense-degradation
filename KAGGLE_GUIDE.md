# Kaggle T4×2 双卡训练指南

在 Kaggle 上跑深度学习，核心逻辑是：**在网页端写好脚本 → 丢给后台服务器跑（Commit） → 跑完后提取 Output 续训。**

文档更新于 2026-09-05；最后一次远端核验为 2026-09-01。V22 已成功在
33/319 completed epochs 处触发 runtime guard，并非全部训练完成。Session 3
当时因 GPU 配额未能提交；不要把当时“3 天后恢复”的提示当作今天的状态。
先阅读[执行快照](RUN_STATUS_2026-09-05.md)和[冻结协议](CLEAN_HORIZON_BASELINE_PROTOCOL_2026-08-30.md)。
本文示例不替代现有已冻结 Notebook；当前续训不要把 resume 路径清空。

---

## 第一阶段：环境与数据准备

1. **复用现有 DINO 训练 Notebook**：不要新建 Notebook；每轮都在同一
   Notebook 里通过 `Save Version + Run All` 提交。
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
4. **续跑轮挂载上一轮 Notebook Output**：
   - 第一轮只挂载 ImageNet-100，从 epoch 0 开始；
   - 后续轮通过 `+ Add Input` 挂载上一轮成功 Version 的 Output；
   - 只使用其 `dino_clean_horizon_seed0/checkpoint.pth`，不得使用历史
     stitched checkpoint。

---

## 第二阶段：Notebook 代码编写

在 Kaggle Notebook 中，按顺序创建以下 **5 个代码块（Cell）**：

### Cell 1：下载代码

```python
%cd /kaggle/working
!rm -rf /kaggle/working/dino
!git clone https://github.com/xbz123/dino-dense-degradation.git /kaggle/working/dino
%cd /kaggle/working/dino
!git checkout --detach 4c16679e915ca1e84842d652c911166f164b5183
```

> ⚠️ 必须先 `%cd /kaggle/working`，否则删除 dino 文件夹后 shell 会找不到当前目录而报错。

### Cell 2：确认当前代码版本

```python
%cd /kaggle/working/dino
!git rev-parse HEAD
!git status --porcelain
!python -m py_compile main_dino.py utils.py clean_horizon_contract.py
```

HEAD 必须精确输出 `4c16679e915ca1e84842d652c911166f164b5183`，且
`git status --porcelain` 必须为空。Notebook 中不得临时改源码。

### Cell 3：设置断点续训路径

```python
import os

os.environ['CLEAN_HORIZON_REPO_DIR'] = '/kaggle/working/dino'
os.environ['CLEAN_HORIZON_OUTPUT_DIR'] = '/kaggle/working/dino_clean_horizon_seed0'
os.environ['CLEAN_HORIZON_DATA_PATH'] = '/kaggle/input/datasets/wilyzh/imagenet100/ImageNet100/train'
os.environ['CLEAN_HORIZON_VAL_PATH'] = '/kaggle/input/datasets/wilyzh/imagenet100/ImageNet100/val'

# 第一轮必须留空。后续轮只填上一轮成功 Version 的 rolling checkpoint。
os.environ['CLEAN_HORIZON_RESUME_FROM'] = ''
print('Resume:', os.environ['CLEAN_HORIZON_RESUME_FROM'] or 'fresh epoch 0')
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

!bash run_clean_horizon_kaggle.sh
```

**参数说明：**
- `--nproc_per_node=2`：使用 2 张 T4 GPU 并行训练
- `--batch_size_per_gpu 64 × 2卡 × --accum_steps 2 = 256` 等效批次大小
- `--epochs 319`：固定一个 schedule，零基 label 范围为 `0..318`
- `--drop_incomplete_accumulation true`：每 epoch 使用 988 个 micro-batches、
  494 个完整 optimizer steps，不执行第 989 个不完整 accumulation group
- `--resume_from`：只读取上一轮同一 clean-horizon run 的 rolling checkpoint
- `--saveckp_freq 10`：每 10 个 epoch 保存一次历史 checkpoint
- `--keep_last_ckpts 0`：保留所有历史 checkpoint，便于后续 dense degradation sweep
- `--milestone_ckpt_epochs 180 250 318`：强制保留正式评测点
- `--local_crops_number 4`：冻结配方使用 4 个局部裁切，不在续训时更改
- `--diag_every 5`：每 5 个 epoch 运行一次稠密退化诊断
- runtime guard 会在完整 epoch/checkpoint 后退出，为 Kaggle 封存预留时间
- 显存占用约 9.6 GB / 16 GB，每 epoch 约 40 分钟

---

## 第三阶段：一键后台挂机（Commit）

正式训练通过保存版本后台执行，不依赖浏览器保持打开：

1. 点击 Kaggle 页面右上角的 **`Save Version`**
2. 选择 **`Save & Run All (Commit)`**，点击 `Save`
3. **放心关闭电脑**。Kaggle 会在后台连续跑最多 12 个小时
4. 成功结束后检查该 Version 的 Output；不能假设异常或平台超时一定保留
   完整产物，也不要将云端 Output 当作唯一永久备份。

---

## 第四阶段：收获结果与开启下一轮循环

Version 结束后：

1. 在 **`Output`** 中检查
   `dino_clean_horizon_seed0/clean_horizon_session_summary.json`。状态必须是
   `partial_runtime_guard` 或 `complete`，rolling checkpoint 的记录大小必须
   与 `dino_clean_horizon_seed0/checkpoint.pth` 一致。
2. **开启下一轮**：
   - 点击 `Edit` 回到编辑界面
   - 点击右侧 `+ Add Input` → 选择 `Your Work` → 挂载你**自己上一轮的 Notebook**
   - 如果同一 Notebook 输入已经挂载，使用 `Check for updates` 并核对确切
     来源 Version；路径字符串不变不代表 checkpoint 版本没变
   - 把 Cell 3 的 `CLEAN_HORIZON_RESUME_FROM` 改成上一轮挂载路径下的
     `dino_clean_horizon_seed0/checkpoint.pth`
   - 再次 `Save Version` → `Save & Run All`，开始新的 12 小时挂机

不得从任意历史 `checkpoint0180.pth`、stitched run 或其他源码版本继续。
每一轮只接受结构化 contract 完全一致且可加载的 rolling checkpoint。
同时检查内部 epoch、attempted/applied 更新数、AMP 累计与连续计数、两份
rank RNG 状态及源码 clean 状态。先独立下载验收，日志和 summary 按 session
分别保存。配额不足时保留草稿，不切 CPU、不改 batch/seed/epoch；提交被拒绝
不能记作新 Version 已启动。

---

## 第五阶段：COCO-Stuff selected-checkpoint 验证

本节为条件式评测参考，当前不可直接执行。历史三 probe-seed teacher VOC
v2 重测已经完成，但其 `epochs=200 / 300 / 500` 的 stitched 轨迹只能用于
探索性刻画，不能开启正式 COCO 判定门。先完成 clean-horizon V2 训练及
其 VOC 科学判定，再对同一 clean-baseline 的 labels 180/318 做 COCO
frozen-backbone linear probing。历史 v1/v2 和 clean V2 数据不能混表；历史
独立 session logs 的补档也不能改变已经观察到的 stitched 分类。

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
   - 挂载包含已验收 clean-baseline checkpoint 的输入，不使用历史 dinockp。
   - 正式 COCO 一致性验证目录里至少要有：
     ```text
     checkpoint0180.pth
     checkpoint0318.pth
     ```
   - VOC 主门另需 `checkpoint0250.pth`；clean 审计保留从 V2 起点开始的
     session 合同、checkpoint 元数据及独立日志。历史
     `REVIEW_BASELINE_2026-07-26.md` 的补档清单不是 clean 父输入清单。
   - 示例路径：
     ```text
     <accepted-clean-baseline-checkpoints>/
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
       cocostuff-10k-v1.1/annotations/*.mat
       cocostuff-10k-v1.1/imageLists/*.txt
     ```

4. **已有 VOC/DSE 结果（可选但推荐）**
   - 如果要生成 `coco_voc_dse_comparison_global_confusion_v2.csv`，挂载
     相同 probe seed、相同 checkpoint key 的 v2 VOC 和 raw/L2 输出：
     ```text
     voc_all_checkpoints/voc_miou_results_global_confusion_v2.json
     figures/combined_dense_summary.csv
     ```
   - `voc_miou_results.json` 是历史 batch-mean-v1 结果，不能作为该比较的
     输入。

### Smoke test：先只跑 180 和 318

先确认数据路径、mask label、nearest-neighbor resize、linear head 训练和 mIoU 计算都正常：

```python
%cd /kaggle/working/dino

!python eval_coco_stuff_dense.py \
    --ckpt_dir '<accepted-clean-baseline-checkpoints>' \
    --coco_root /kaggle/input/datasets/dntai2/cocostuff-10k-v1-1 \
    --epochs 180,318 \
    --img_size 336 \
    --batch_size 64 \
    --train_epochs 1 \
    --lr 0.0025 \
    --loss_resolution patch \
    --feature_dtype float16 \
    --probe_seed 42 \
    --checkpoint_key teacher \
    --max_train_images 2000 \
    --max_val_images 500 \
    --output_dir /kaggle/working/dino_clean_horizon_seed0_eval/coco_stuff_selected_smoke
```

Smoke test 通过后，检查输出：

```text
coco_stuff_miou_results_global_confusion_v2.json
dense_degradation_coco_stuff_global_confusion_v2.png
coco_stuff_summary_global_confusion_v2.md
```

只有同时提供匹配的 `--voc_results_json` 时，才会额外生成
`coco_voc_dse_comparison_global_confusion_v2.csv`。

### 正式 selected-checkpoint run

按 v10b gate，正式一致性验证只跑预注册的 `180 / 318`。更宽的 6-checkpoint
曲线只能作为探索性附加结果，不能替换该配对比较：

```python
%cd /kaggle/working/dino

!python eval_coco_stuff_dense.py \
    --ckpt_dir '<accepted-clean-baseline-checkpoints>' \
    --coco_root /kaggle/input/datasets/dntai2/cocostuff-10k-v1-1 \
    --epochs 180,318 \
    --img_size 336 \
    --batch_size 64 \
    --train_epochs 15 \
    --lr 0.0025 \
    --loss_resolution patch \
    --feature_dtype float16 \
    --probe_seed 42 \
    --checkpoint_key teacher \
    --voc_results_json '<matching-clean-voc-v2-json>' \
    --dense_summary_csv '<matching-clean-structural-summary-csv>' \
    --output_dir /kaggle/working/dino_clean_horizon_seed0_eval/coco_stuff_selected_seed42_teacher
```

对 `1337` 和 `2027` 分别重复正式命令，并使用 seed 专属输出目录。比较器
会 fail closed：VOC 与 COCO 的 `metric_version` 必须都是
`global_confusion_v2`，且 `probe_seed`、`checkpoint_key`、representation、
checkpoint structured identity 和源码 commit/dirty 状态必须完全匹配。每行还必须保留
probe 配方和数据身份，并满足 `source_dirty=false`。

上述占位路径必须替换为已核验输入，配方须与届时获准的 COCO 执行计划
一致。Smoke test 不计入正式结果。COCO 只报告固定 labels 318−180 的
配对差及三 seed 变化，作为次要一致性证据，不改变 VOC 判定、不宣称
180 是全程峰值，也不按结果增加 checkpoint 或替换 seed。

---

## 性能参考

| 平台 | GPU | 步数/epoch | 每 epoch 耗时 |
|------|-----|-----------|-------------|
| Colab Free | T4 ×1 | 1979 | ~66 分钟 |
| **Kaggle Free** | **T4 ×2** | **989** | **~40 分钟** |

- V22 的 15 epochs 平均约 40.9 分钟/epoch，仅为该 session 的观测。
- 若按该速度外推，33 后剩余 286 epochs 约需 195 小时；这不是运行预算
  承诺，实际受配额、I/O、诊断和 session 开销影响。
- 当前冻结合同要求 Kaggle T4 x2；不得直接与 Colab 单卡交替续训。
