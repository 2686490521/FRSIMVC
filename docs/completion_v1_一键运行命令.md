# Completion v1（大纲 v9 · Module III）一键运行命令

生成时间：2026-09-22　服务器：`10.33.7.232`（ubuntu / 11222）
代码目录：`/home/ubuntu/Tangzl/FRSIMVC/FRSIMVC/FRSIMVC-code`
已验证：单元自检 48/48；六数据集冒烟 6/6 任务、48 条臂记录、`missing_runs.txt` 为空；
只读适配器对训练轨迹零侵入（开/关补全的基线逐轮 ACC 完全相同）。

---

## 1. 主实验（阶段 B）：6 数据集 × 10 seed × 300 轮

```bash
ssh -p 11222 ubuntu@10.33.7.232
cd /home/ubuntu/Tangzl/FRSIMVC/FRSIMVC/FRSIMVC-code
mkdir -p results/completion_v1/_launches
LOG="results/completion_v1/_launches/nohup_$(date +%Y%m%d_%H%M%S_%N).log"
nohup env GPUS_STR="2 3 4 5 6 7" RUN_TAG=v1 bash run_completion_sweep.sh > "$LOG" 2>&1 &
tail -f "$LOG"
```

- 训练任务 60 个，臂记录 480 条；预计 **1.5–3 h**（Trento 300 轮约 5–6 min，大数据集 2–3×）。
- 断 SSH 不影响：`nohup` + 独立日志。查进度：`tail -f "$LOG"` 或 `cat results/completion_v1/batch_*/missing_runs.txt`。
- 脚本自带：nvidia-smi 选卡（剔除 <10 GB 空闲）、每卡 1 任务、OOM 自动换最空闲卡重试 2 次、结束完整性审计。

## 2. 跑之前建议先做的两步（各 1 分钟内）

```bash
cd /home/ubuntu/Tangzl/FRSIMVC/FRSIMVC/FRSIMVC-code
DRY_RUN=true bash run_completion_sweep.sh          # 只看会提交什么命令
VALIDATE_ONLY=true bash run_completion_sweep.sh    # 6 个数据集各 1 轮：数据路径 + 冻结基线断言
```

## 3. 常用变体

```bash
# 冒烟：6 数据集 × seed0 × 3 轮 → results/completion_v1_smoke
ONLY_SMOKE=true bash run_completion_sweep.sh

# 小批试跑
DATASETS_STR='Trento MUUFL' SEEDS_STR='0 1' GPUS_STR='2 3' bash run_completion_sweep.sh

# 自定义诊断轮次（伪缺失探针触发点，默认 100/200/300）
DIAGNOSTIC_EPOCHS_STR='100 200 300' EPOCHS=300 bash run_completion_sweep.sh

# 阶段 C 确认批：换 seed 10-19，λ 冻结
SEEDS_STR='10 11 12 13 14 15 16 17 18 19' RUN_TAG=confirm bash run_completion_sweep.sh

# 指定显卡 / 更少卡
GPUS_STR="4 5" bash run_completion_sweep.sh
```

## 4. 结果在哪

```
results/completion_v1/batch_<时间戳>_<tag>/
├── manifest.json            实验契约（含 lambdas/arms/seed/冻结基线参数）
├── environment.json         版本与环境变量
├── source_sha256.json       代码指纹
├── gpu_preflight.txt        选卡记录
├── missing_runs.txt         为空才算完整
├── <Dataset>/seed_<k>/{completion_summary.json, arms/<ARM>/metrics.json, diagnostics/*}
└── _analysis/{completion_runs.json, report.md, completion_summary.csv}
```

八臂：`B0`（冻结基线）`E0`（刷新聚类头 + 按 m 融合）`G0`（+ 缺失视图自身图预测）
`D0`（+ donor 分布 q）`C025/C050/C075/C100`（λ=0.25/0.5/0.75/1.0 的语义—图凸组合）。

## 5. 判读口径（必须遵守）

1. 主判据 = 同一条轨迹内 `C050 − B0` 的配对差（60 对），非劣阈值 δ=0.005。
2. 主表并列 **best / median / tail30 / final + 逐轮 std**；只报 best 会偏袒高方差一方。
3. 八臂共享一条训练轨迹，**不能**当作 8 次独立训练计数。
4. 跨 run 绝对数不可比：实测同配置重复两次逐轮 ACC 最大差 0.058（即便开 `--deterministic-mode`）。
5. 已知结构事实（写论文时要说明）：本批数据 `eligible_rate == missing_rate`、`removed_mass = 0` ⇒ **D0 ≡ E0 恒等**；
   伪缺失探针显示 NMSE 随 λ 单调上升（G0 0.0020 → C100 0.0293）⇒ 补全收益在聚类层而非逐点重建层；
   `‖h_sem‖ ≈ 2.6 × ‖h_graph‖` 且 cosine ≈ 0.99 ⇒ λ 同时携带"语义强度 + softmax 锐度"两种效应。
