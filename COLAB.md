# 在 Google Colab 上训练 MiniMind

> 本文件是针对本 fork（`Ezra-Maker-MAX/minimind`）的 **Colab 训练笔记**：目录规划、四阶段路线、实测踩坑与排障。
>
> ⚠️ **状态说明（务必先读）**：本文档的结论来自 **静态代码核对 + 官方文档推算**，**尚未在真实 Colab 实例上完整跑通验证**。
> 三处硬约束（Python 3.13 wheel / T4 无 bf16 / checkpoint 路径硬编码）均已通过读源码确认，但端到端时长与稳定性仍需自测。
> 实测完成后请把本节改为「已验证」并附上日志链接。

---

## 结论

**可以跑**，但上游 README 的默认配置在 Colab 上会踩三个坑，不改必然失败或严重掉速。改完之后，64M 的 `minimind-3` 在免费层 T4 上可以完成 pretrain + SFT。

**但更重要的是选对阶段**——免费层的真正甜区不是 pretrain，而是 P2/P3（数据量小、单次会话可跑完）。

---

## 一、四阶段路线（按能力递进，而非按文件）

| 阶段 | 做什么 | 需要数据 | 免费层可行性 |
|---|---|---|---|
| **P0 冒烟** | 200 行假数据跑通一个完整 step | 无 | ✅ 约 5 分钟 |
| **P1 基座** | pretrain → SFT | mini 组合 2.98 GB | ⚠️ 6h+，必须分多次会话 |
| **P2 对齐** | DPO（可选 GRPO / Agent） | dpo + rlaif + agent_rl ≈ 178 MB | ✅ 单次会话可跑完 |
| **P3 定制** | LoRA（医疗 / 身份） | lora_* 三文件 ≈ 59 MB | ✅ 最快见效 |

**建议顺序**：P0 → P2 → P3 → P1。
理由：P2/P3 数据加起来不到 300MB，一次会话就能拿到可对比的结果；P1 那 6 小时随时可能被回收，适合当作长线任务。

---

## 二、三个必须规避的坑

### 坑 1：不要 `pip install -r requirements.txt`

`requirements.txt` 钉死了 `numpy==1.26.4`。Colab 是 **Python 3.13**，这个版本没有 cp313 预编译 wheel：

```
$ pip download numpy==1.26.4 --only-binary=:all: --python-version 3.13
ERROR: Could not find a version that satisfies the requirement numpy==1.26.4
       (from versions: 2.1.0 ... 2.2.6)
```

更糟的是 pip 会为这个死版本做**长时间依赖回溯**，把 kernel 占死（实测卡 20 分钟以上，后续 cell 全部排队，且 colab-mcp 没有 interrupt 工具）。

✅ **正确做法**——只装训练必需子集，numpy 保持 Colab 自带的 2.1.x：

```bash
!pip -q install "numpy>=2.1" datasets transformers einops rich huggingface_hub \
    jsonlines datasketch simhash psutil jieba nltk scikit_learn trl wandb swanlab
```

**这份清单是核对 `requirements.txt` + 逐个 grep 训练脚本 import 后得出的**，比"只装 6 个包"的常见写法稳妥。逐个说明：

| 包 | 被谁需要 | 漏装后果 |
|---|---|---|
| `datasets` | 全部训练脚本 | 直接崩 |
| `transformers` | 全部 | 直接崩 |
| `einops` | `model_minimind.py` | 直接崩 |
| `rich` | `trainer_utils.Logger` | 直接崩 |
| `huggingface_hub` | 数据下载 | 直接崩 |
| `jsonlines` | `dataset/lm_dataset.py` | 部分数据集加载崩 |
| `tr`l | 对齐相关 | 部分脚本崩 |
| `wandb` / `swanlab` | `trainer_utils` 内 import | **即使不用也会崩** |
| `datasketch` / `simhash` | 数据去重工具 | 用到才崩 |
| `psutil` / `jieba` / `nltk` / `scikit_learn` | 辅助脚本 | 用到才崩 |

> `torch` 在 requirements.txt 里本来就被注释掉，直接用 Colab 预装的 GPU 版。

### 坑 2：T4 没有 bfloat16，必须 `--dtype float16`

训练脚本默认 `--dtype bfloat16`，但 T4 是 **sm_75**，bf16 需要 sm_80+ 才有硬件支持。

✅ 按架构自动选：

```python
import torch
p = torch.cuda.get_device_properties(0)
DTYPE = 'bfloat16' if p.major >= 8 else 'float16'   # T4 → float16
```

只有拿到 A100/H100 才用 bfloat16。

### 坑 3：中间产物不落 Drive 就白训

Colab 的 `/content` 是**临时盘**，实例一释放全没。而续训需要的 optimizer state / scaler / epoch / step 都在 checkpoint 里。

⚠️ **特别注意**：checkpoint 路径是**硬编码**的（已核查 8 个训练脚本全部如此）：

```python
lm_checkpoint(..., save_dir='../checkpoints')   # --save_dir 管不到它！
```

`--save_dir` 只控制模型权重输出目录（默认 `../out`），**管不了这个 resume 检查点**。

✅ **最省事的解法：把整个项目放 Drive**，让这个硬编码路径天然落在持久存储上：

```bash
git clone --depth 1 https://github.com/Ezra-Maker-MAX/minimind.git /content/drive/MyDrive/minimind
```

不用软链，也不用改源码。

---

## 三、目录规划

```
/content/drive/MyDrive/minimind/     ← 项目（持久）
├── checkpoints/    ← 断点续训状态：optimizer / scaler / epoch / step
├── out/            ← 模型权重 *.pth
├── logs/           ← train.log + last_cmd.sh（启动命令存档）
└── dataset/        ← 数据集（持久，避免每次重下）

/content/minimind/dataset/           ← 每次会话的工作副本（VM 本地盘，训练读这里）
```

**为什么数据集要 copy 到本地盘**：Drive 走 FUSE，训练时 dataloader 反复读会被拖慢；VM 的 `/content` 是本地 NVMe。用 `shutil.copy2` 保留 mtime，还能让 `datasets` 复用 arrow 缓存。

---

## 四、数据集真实体积（实测，非 README 标称）

| 文件 | 实测大小 | 用于 | 阶段 |
|---|---|---|---|
| `pretrain_t2t_mini.jsonl` | 1.24 GB | `train_pretrain.py` | P1 |
| `sft_t2t_mini.jsonl` | 1.74 GB | `train_full_sft.py` | P1 |
| `dpo.jsonl` | 53.7 MB | `train_dpo.py` | P2 |
| `rlaif.jsonl` | 23.8 MB | `train_grpo.py` | P2 |
| `agent_rl.jsonl` | 82.0 MB | `train_agent.py` | P2 |
| `agent_rl_math.jsonl` | 18.4 MB | `train_agent.py`（数学） | P2 |
| `lora_exam.jsonl` | 24.7 MB | 基座评测 | P3 |
| `lora_identity.jsonl` | 22.8 KB | `train_lora.py`（身份） | P3 |
| `lora_medical.jsonl` | 34.0 MB | `train_lora.py`（医疗） | P3 |
| `pretrain_t2t.jsonl` | **8.28 GB** | 完整版 | ❌ |
| `sft_t2t.jsonl` | **14.10 GB** | 完整版 | ❌ |

**关键数字**
- mini 组合 = **2.98 GB**（非 README 说的 2.8GB）
- 完整版组合 = **22.4 GB** → 免费 Drive 15GB **装不下**
- **P2 + P3 全套 = 236 MB** → 免费层毫无压力，**这是最被低估的部分**

数据源：`jingyaogong/minimind_dataset`（**注意是 `_dataset` 后缀**，写成 `jingyaogong/minimind` 会全 404）

---

## 五、快速开始

```bash
# 0) Colab 菜单：修改 → 笔记本设置 → 硬件加速器 → T4 GPU → 保存

# 1) 装依赖（绝不要 -r requirements.txt）
!pip -q install "numpy>=2.1" datasets transformers einops rich huggingface_hub \
    jsonlines datasketch simhash psutil jieba nltk scikit_learn trl wandb swanlab

# 2) 挂 Drive + 项目放 Drive
from google.colab import drive; drive.mount('/content/drive')
!git clone --depth 1 https://github.com/Ezra-Maker-MAX/minimind.git /content/drive/MyDrive/minimind

# 3) 一次性灌数据到 Drive（已存在则跳过）
!mkdir -p /content/drive/MyDrive/minimind/dataset
from huggingface_hub import hf_hub_download
for fn in ['pretrain_t2t_mini.jsonl', 'sft_t2t_mini.jsonl']:
    hf_hub_download(repo_id='jingyaogong/minimind_dataset', filename=fn,
                    repo_type='dataset', local_dir='/content/drive/MyDrive/minimind/dataset')

# 4) 同步到本地盘
!mkdir -p /content/minimind/dataset
!cp -p /content/drive/MyDrive/minimind/dataset/*.jsonl /content/minimind/dataset/

# 5) 预训练（数据路径用绝对路径，别用 ../dataset）
%cd /content/drive/MyDrive/minimind/trainer
!python train_pretrain.py --epochs 1 --batch_size 16 --accumulation_steps 8 \
    --num_workers 2 --dtype float16 --use_compile 1 \
    --data_path /content/minimind/dataset/pretrain_t2t_mini.jsonl \
    --log_interval 50 --save_interval 500 --from_resume 1
```

---

## 六、提速：`--use_compile` 是被漏掉的杠杆

`train_pretrain.py:106` / `train_full_sft.py:107` / `train_lora.py:101` 都有 `--use_compile`，**默认 0（关闭）**，上游文档从未提及。

T4 算力只有 3090 的 40~50%，pretrain 有 2.4 万+ step。开启后的收益：

```bash
--use_compile 1    # 追加到训练命令
```

注意：
- 首次 step 会有 **2~5 分钟编译开销**（把整轮时间算进去仍划算）
- MCP 驱动时这个编译期**会超过单次 cell 的默认超时**，必须用后台进程 + 日志轮询的方式启动（`colab_runner.py` 已如此设计）
- 若遇到 compile 相关报错，直接去掉该参数回退，不影响正确性

**其他可调项**

| 参数 | 默认 | 建议 | 说明 |
|---|---|---|---|
| `--num_workers` | 8 | 2 | 免费层只有 2~4 核，开多更慢 |
| `--save_interval` | 1000 | 500 | 缩短以降低断点损失 |
| `--batch_size` | 32 | 16 | T4 显存够，但大 batch 未必更快 |
| `--max_seq_len` | 340 | 不变 | 想省时间可降到 256（影响效果） |

---

## 七、阶段详解

### P0 冒烟（先跑这个）

用 200 行假数据验证 forward / backward / 保存全链路，**不要一上来就烧 6 小时**。

```bash
python colab_runner.py smoke
```

看到 loss 数值即代表链路打通。

### P2 对齐（性价比最高）

**DPO**——不需要额外模型，直接可跑：

```bash
python train_dpo.py --epochs 1 --batch_size 4 --accumulation_steps 2 \
    --dtype float16 --data_path /content/minimind/dataset/dpo.jsonl \
    --from_weight full_sft --from_resume 1
```

⚠️ **GRPO / Agent 的隐藏门槛**：这两个脚本需要 **Reward Model**（`trainer_utils.LMForRewardModel`），默认路径 `../../internlm2-1_8b-reward`。

- 需额外下载 `internlm/internlm2-1_8b-reward`（约 3.6 GB）
- 下载后放到项目上一级目录，或改 `--reward_model_path`
- **这部分未验证**；免费 Drive 15GB 下与 mini 数据同时存在会吃紧

### P3 定制（最快见效）

```bash
# LoRA 医疗
python train_lora.py --lora_name lora_medical --epochs 3 --batch_size 16 \
    --dtype float16 --data_path /content/minimind/dataset/lora_medical.jsonl \
    --from_weight full_sft --from_resume 1

# LoRA 身份
python train_lora.py --lora_name lora_identity --epochs 10 --batch_size 16 \
    --dtype float16 --data_path /content/minimind/dataset/lora_identity.jsonl \
    --from_weight full_sft --from_resume 1
```

---

## 八、断点续训

Colab 免费层会话随时可能被回收，**一切以"能接着跑"为前提设计**：

1. 产物全在 Drive（见第三节），实例释放不丢
2. 训练命令统一带 `--from_resume 1`，重跑自动加载 `checkpoints/`
3. 启动命令存档到 `logs/last_cmd.sh`，保证续训参数与上次一致
4. 日志同步到 Drive，新会话能看到历史 loss

被中断后重新运行**同一条命令**即可。

⚠️ Colab **没有外部唤醒通道**，实例被回收后无法自动重启，这一步必须人工触发。

---

## 九、时长预期

README 标称基于单卡 3090：pretrain_mini ≈1.21h/epoch、sft_mini ≈1.10h/epoch。
**T4 约为 3090 的 40~50% 算力**，换算：

| 阶段 | T4 预估（1 epoch） |
|---|---|
| pretrain（mini） | **2.5 ~ 3 小时**（开 compile 可降） |
| SFT（mini） | **2.5 ~ 3 小时** |
| DPO | 视数据量，**分钟级** |
| LoRA（医疗/身份） | 视数据量，**分钟到十分钟级** |

**显存不是瓶颈**：64M 模型训练时参数+梯度+AdamW 约 1GB 出头，T4 的 16GB 绰绰有余。瓶颈是算力。

---

## 十、自动化脚本

仓库里的 `tools/colab_runner.py` 封装了全流程：

```bash
python colab_runner.py smoke                       # 冒烟
python colab_runner.py prepare --data p2           # 一次性灌 P2 数据到 Drive
python colab_runner.py pretrain --skip-prepare --watch 120
python colab_runner.py sft      --skip-prepare --watch 120
python colab_runner.py dpo      --skip-prepare --watch 30
python colab_runner.py lora     --lora-name lora_medical --skip-prepare --watch 30
python colab_runner.py status                      # 重连看日志
```

支持的 `--data` 组合：`mini`（2.98GB）/ `full`（22.4GB，装不下）/ `p2`（对齐 178MB）/ `p3`（LoRA 59MB）/ `rl`（兼容旧名）

设计要点：
- 训练用 `start_new_session=True` 后台起 + 日志落盘
- 每次看日志顺手备份到 Drive
- 启动前 `pgrep` 检查，避免重复起进程
- 唯一需要人工的一步：浏览器点「连接」

---

## 十一、排障

| 现象 | 原因 / 处理 |
|---|---|
| 所有 cell 返回空、`execution_count` 为 null | kernel 被前序 cell 占死（常见于 pip 依赖回溯）→ Colab UI 点停止 |
| `ModuleNotFoundError: rich/wandb/jsonlines` | 依赖清单没装全 → 见坑 1 的完整清单 |
| `CUDA out of memory` | 调小 `--batch_size`，或加 `--accumulation_steps` 补偿 |
| loss 不降 / NaN | T4 上忘改 dtype → 加 `--dtype float16` |
| 重跑后从头开始 | 没带 `--from_resume 1`，或 `checkpoints/` 不在 Drive 上 |
| 训练极慢 | 数据集在读 Drive 而非本地盘 → copy 到 `/content` 再用 |
| 开 `--use_compile` 后长时间无输出 | 首次编译需 2~5 分钟，属正常；超过 10 分钟检查日志 |
| GRPO/Agent 报 reward model 找不到 | 需额外下载 `internlm2-1_8b-reward`，见 P2 说明 |
| `flash_attn` 相关报错 | 无需担心，项目用的是 PyTorch 内置 `F.scaled_dot_product_attention` |
| 下载数据集 404 | 仓库名是 `jingyaogong/minimind_dataset`（带 `_dataset` 后缀） |

---

## 十二、尚未验证的部分（诚实清单）

- ❌ **端到端未在真实 Colab 上跑过**——本文所有结论来自源码核对与文档推算
- ❌ `--use_compile` 在 T4 + PyTorch 2.11 下的实际加速比
- ❌ GRPO / Agent 的 reward model 下载与显存占用
- ❌ A100 会话下的 bfloat16 路径（免费层多数时候只给 T4）
- ❌ 完整版 22.4GB 数据集（受限于免费 Drive 15GB，未测）

---

## 十三、上游同步

本 fork 与上游 `jingyaogong/minimind` 保持代码同步。合入方式（因本仓库是独立建仓而非 GitHub fork 关系，用 API 合）：

```bash
# 拉取上游最新提交并合入
curl -X POST -H "Authorization: Bearer $TOKEN" \
  https://api.github.com/repos/Ezra-Maker-MAX/minimind/merge-upstream \
  -d '{"branch":"master"}'
```

> 注：`merge-upstream` 仅在仓库被 GitHub 识别为 fork 时可用；本仓库为独立建仓，需用 `git remote add upstream` + `git merge` 本地合并后推送。

---

*基于 2026-09 静态核对整理。Colab 政策与镜像环境变化较快，若流程跑不通请先看第十一节。*
