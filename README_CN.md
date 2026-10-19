# 使用说明

代码和命令统一使用 `site-1`（工地一）、`site-2`（工地二）。公开图片继续使用 `工地一_0001`、`工地二_0001`；转换后的工作目录为 `data/site-1`、`data/site-2`。

## 当前交付：仅公开 500 张

当前上传目录为 `public_data/`，工地一 100 张、工地二 400 张，使用中文文件名，每张都有分割、边缘、中心标注和实测表。见 `public_data/README.md`。完整数据、完整标注表和旧全量清单已移出本发布目录。

运行代码前先执行 `python scripts/prepare_public_data.py`，生成被 Git 忽略的 `data/` 和 `annotation_all_images/` 兼容目录。公开划分为工地一训练 73/验证 17，工地二训练 317/验证 79，另有 14 张保留为未分配。下面提到的 302/69 和 519/130 是原论文全量划分，不随此公开包提供；用公开子集重跑不能宣称复现了全量论文结果。

这是独立整理的代码和完整数据目录，原工程未被替换。所有命令在本目录根下执行。

## 数据

- 转换后的 `data/site-1`：公开工地 1，训练 73 张、验证 17 张、未分配 10 张。
- 转换后的 `data/site-2`：公开工地 2，训练 317 张、验证 79 张、未分配 4 张。
- 数据目录可能还含不在当前划分中的图片；实验只读取 `gt_sets/train.txt` 和 `val.txt`，不会自动纳入额外图片。
- `images` 是原图，`segmentation` 是人工分割标注，`edge` 是边缘标注，`rebar_labels` 是中心位置标注。
- 历史文件名 `depth_labels.txt` 存的是激光标量距离，单位米，不是深度图。
- `annotation_all_images/all_images_annotations.csv` 中 `submitted_spacing_mm` 是用户核查过的读数；`image_reference_mm` 等草稿参考列不是物理真值。
- 设计间距和有符号上下偏差来自现有区域标注。不要为了提高指标调整这些值。

## 环境与执行顺序

推荐 GPU 训练使用 Ubuntu 20.04/22.04、Python 3.10、PyTorch 2.2.2 + torchvision 0.17.2。核心依赖可先安装 PyTorch，再安装 `requirements_external.txt`。不含模型权重，首次运行骨干可能联网下载；正式训练前先检查网络和 GPU。

```bash
conda create -n rebar python=3.10 -y
conda activate rebar
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_external.txt
```

1. 本方法和跨工地：`python scripts/run_revision_experiments.py --run_root runs/revision_full_seed42 --variants full --seeds 42 --execute`。两个工地分别训练，并做双向迁移，不混合训练。
2. 外部模型：`python scripts/run_external_experiment_one.py --methods deeplab gscnn sctransnet afenet --sites site-1 site-2`。
3. 消融：`python scripts/run_experiment_three.py`。失败修复后加 `--resume`，不需要重新开始成功的阶段。
4. 报告生成：按英文 README 的 `run_revision_followups.py 4` 步骤，先训练 QAEP 和模板评估，再设置自己的 API key、加 `--run-api`。不要把密钥写入代码。
5. 筛查一致性：第一步完成后执行 `python scripts/run_screening_agreement.py`。只需 CPU，输出逐间距、逐图 CSV 和汇总 JSON。未附旧预测，因此全新目录不能跳过第一步直接复现此实验。

默认训练为 50 轮、seed 42；每个数据集分别使用训练集校准峰值，验证集只评估。图像匹配覆盖率和漏判率必须与条件误差一起报告。

## Mobile-Seed

Mobile-Seed 使用独立的 Python 3.9 / PyTorch 1.12.1+cu113 环境，不能直接装进主环境。参考已使用的兼容依赖：

```bash
conda create -n rebar_mobile python=3.9 -y
conda activate rebar_mobile
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install numpy==1.23.5 scipy==1.10.1 Pillow==9.5.0 matplotlib==3.7.5 opencv-python==4.8.1.78 scikit-image==0.21.0 timm==0.6.13 einops==0.6.1 prettytable terminaltables packaging==23.2 yapf==0.40.1 addict requests
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
pip install -e baselines_clean/official/Mobile-Seed/repo --no-deps
```

从作者仓库 https://github.com/WHU-USI3DV/Mobile-Seed 的权重链接下载 `AFFormer_tiny_ImageNet1k.pth`，放入 `baselines_clean/official/Mobile-Seed/repo/ckpt/`。

```bash
python -u scripts/run_mobile_seed_experiment_one.py --sites site-1 site-2
```

预训练权重不在本目录中。其他方法使用的本地权重路径和下载行为见对应官方源码及适配器；不得把缺失权重的随机初始化结果标为相同预训练设置。

## GitHub 上传

公开数据位于 `public_data/`，不要使用之前含全量数据的发布副本。已经提供 `.gitattributes`，用 Git LFS 跟踪公开图片；源代码和划分文本正常用 Git 管理。Git LFS 存储/流量额度由账号决定，也可以将该公开子集作为独立 Release 资产提供。

```bash
git init
git lfs install
git add .
git status
```

确认文件清单后自行 commit 和 push。本次只整理本地目录，没有创建远端仓库、提交或推送。`.gitignore` 排除 API 密钥、权重、运行结果和缓存。第三方源码保留原许可证，本项目未擅自指定新的开源许可证。

## 验证范围

`python scripts/check_release_data.py` 检查数据配对、划分、距离与间距标注。`python -m unittest discover -s tests` 运行单元测试。整理发布目录不会重新训练所有模型；GPU 训练和各官方环境仍需要在对应机器执行。
