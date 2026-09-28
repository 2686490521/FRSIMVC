# FRSIMVC 联邦缺失多视图遥感聚类实验

本仓库整理了截至 **2026 年 9 月 28 日**可核实的实验代码和研究记录。最近一批已完成的训练是 2026 年 9 月 22 日结束的 Completion v1：六个数据集、10 个 seed、300 轮，共 60 条共享训练轨迹和 480 条评估臂记录。新补全方案的预设主臂 C050 未达到替换基线的条件；当前正式参照仍是固定图融合的 F05 + BSSAT-r1。

## 从哪里开始

| 内容 | 位置 |
| --- | --- |
| 当前状态、具体效果、失败模块、下一步 | [项目进展与结果](docs/项目进展与结果_20260928.md) |
| 服务器最新完整实验代码 | [FRSIMVC-code](FRSIMVC-code/) |
| Completion v1 一键脚本 | [run_completion_sweep.sh](FRSIMVC-code/run_completion_sweep.sh) |
| 最新六数据集原始汇总 | [Completion 批次](FRSIMVC-code/results/completion_v1/batch_20260922_193907_452645837_1242264_v1/_analysis/) |
| 完整实验配置和源码指纹 | [manifest.json](FRSIMVC-code/results/completion_v1/batch_20260922_193907_452645837_1242264_v1/manifest.json)、[source_sha256.json](FRSIMVC-code/results/completion_v1/batch_20260922_193907_452645837_1242264_v1/source_sha256.json) |
| 历史分析与设计记录 | [docs](docs/) |

## 代码与数据来源

活动代码来自服务器 `/home/ubuntu/Tangzl/FRSIMVC/FRSIMVC/FRSIMVC-code`；进展、历史报告和 v9 大纲来自本地工作文件夹。服务器结果的原始批次目录为：

```text
/home/ubuntu/Tangzl/FRSIMVC/FRSIMVC/FRSIMVC-code/results/completion_v1/
  batch_20260922_193907_452645837_1242264_v1/
```

上传的是此批次中的配置、每 seed 指标与逐轮 CSV、伪缺失诊断以及聚合报告。原始遥感数据、模型 checkpoint、预测张量和大量逐 seed 图片保存在服务器上；仓库保留批次 ID、环境信息及源码哈希，方便追溯。代码快照中纳入与当前实验直接相关的主程序和分析脚本；历史消融启动脚本没有混入默认运行入口。已上传的 39 个代码文件与该批次 `source_sha256.json` 的哈希逐项一致；清单里另有 6 个未收录的历史 RADG 辅助脚本，见[进展报告](docs/项目进展与结果_20260928.md)。

## 运行入口

需要 Linux、NVIDIA GPU、Python 环境和六数据集原始文件。服务器该批次使用 Python 3.8.20、PyTorch 2.4.1+cu118、CUDA 11.8；详情见批次的 `environment.json`。先将 [`config.json`](FRSIMVC-code/config.json) 的 `data_root` 改为自己的数据根目录，并确认 GPU 编号和 Python 可执行文件：

```bash
cd FRSIMVC-code
GPUS_STR="2 3 4 5 6 7" PYTHON_BIN=/path/to/python DRY_RUN=true bash run_completion_sweep.sh
GPUS_STR="2 3 4 5 6 7" PYTHON_BIN=/path/to/python ONLY_SMOKE=true bash run_completion_sweep.sh
GPUS_STR="2 3 4 5 6 7" PYTHON_BIN=/path/to/python bash run_completion_sweep.sh
```

脚本默认跑六数据集、seeds 0–9、300轮。结果自动保存在本方法的 `FRSIMVC-code/results/completion_v1/` 下。可用 `DATASETS_STR`、`SEEDS_STR`、`GPUS_STR`、`EPOCHS`、`MISSING_RATE` 等环境变量调整。已完成批次的数值是历史证据，新跑结果需独立分析，不能与历史不同批次的均值直接当作配对比较。

## 当前判断

Completion v1 的 C050 在六数据集等权 Best ACC 上相对 B0 下降 **0.0267**，五个数据集的均值下降，六个数据集均未通过预设非劣门槛。早期 BSSAT 对 Best ACC 有增益，但其指标与训练后段存在取舍。质量调制与 RADG 的已测版本没有进入正式方法。具体证据和下一步验证项目见[项目进展与结果](docs/项目进展与结果_20260928.md)。

此仓库保存实验状态，不将待验证方案写成已证实的改进。

