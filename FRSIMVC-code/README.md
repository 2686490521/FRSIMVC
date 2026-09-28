# FRSIMVC 代码与批次索引

此目录是服务器活动代码 `/home/ubuntu/Tangzl/FRSIMVC/FRSIMVC/FRSIMVC-code` 的整理快照，关联的已完成实验批次为 `20260922_193907_452645837_1242264`。主入口是 `main.py`，批量入口是 `run_completion_sweep.sh`，聚合分析为 `report_completion.py`，Completion 自检为 `_test_completion.py`。

默认实验协议：Augsburg、MDAS、MUUFL、Salinas、Trento、XuZhou；random@0.3 独立视图缺失并零填充；mask-before-SLIC；A3；固定融合 beta=0.5；Sinkhorn epsilon=0.1；质量调制 alpha=0；BSSAT dual+balanced；RADG 关闭；FedAvg；300 轮；seeds 0–9。每个数据集与 seed 只训练一次，B0、E0、G0、D0、C025、C050、C075、C100 是同一轨迹的八个只读评估臂。

`config.json` 中的数据目录是原服务器绝对路径，移植时需要调整。`run_completion_sweep.sh` 默认的 `PYTHON_BIN` 和 `GPUS_STR` 也对应原服务器；在新机器上通过环境变量覆盖。数据文件未提交。运行前先执行 `DRY_RUN=true bash run_completion_sweep.sh` 与 `ONLY_SMOKE=true bash run_completion_sweep.sh`。

已完成实验的追溯文件位于 `results/completion_v1/batch_20260922_193907_452645837_1242264_v1/`。上传的逐轮 CSV 与 arm 指标足以重算主表；完整模型 checkpoint 和像素图仍在原服务器批次目录。`results/radg_v11/_analysis/` 是旧消融的汇总证据，RADG 并非当前默认模块。

注意：原批次的 `source_sha256.json` 覆盖了活动代码和一些历史诊断脚本。本整理快照省略了 `_calibrate_radg_gamma.py`、`_test_radg.py`、`_test_radg_audit.py`、`_test_report_integrity.py`、`report_radg.py`、`rerun_incomplete_runs.py`；其余收录的39个代码文件哈希全部匹配。详情见仓库根目录的进展报告。
