# 在 Google Colab 上训练 MiniMind

> 本文件是针对本 fork（`Ezra-Maker-MAX/minimind`）的 **Colab 训练专用笔记**，记录实测踩过的坑和可直接复用的流程。
>
> 实测环境：Colab **免费层 T4 (16GB)** / Python **3.13.15** / torch 2.11.0 / 2026-09

## 结论

**能跑**，但官方 README 的默认配置在 Colab 上会踩三个坑，不改必然失败或严重掉速。全部改完之后，64M 的 `minimind-3` 在 T4 上用 mini 数据可以跑完 pretrain + SFT。

---

## 一、三个必须规避的坑

### 坑 1：不要 `pip install -r requirements.txt`

`requirements.txt` 里钉死了 `numpy==1.26.4`。Colab 是 **Python 3.13**，而这个版本没有 cp313 预编译 wheel：

```
$ pip download numpy==1.26.4 --only-binary=:all: --python-version 3.13
ERROR: Could not find a version that satisfies the requirement numpy==1.26.4
       (from versions: 2.1.0 ... 2.2.6)
ERROR: No matching distribution found for numpy==1.26.4
```

更糟的是，pip 会为了这个死版本做**长时间依赖回溯**，把 Colab kernel 占死（实测卡了 20 分钟以上，后续 cell 全部排队无法执行，且 MCP 没有 interrupt 工具只能从 UI 停）。

✅ **正确做法**——只装训练必需子集，numpy 保持 Colab 自带的 2.1.3：

```bash
!pip -q install "numpy>=2.1" datasets transformers einops rich huggingface_hub
```

（`torch` 在 requirements.txt 里本来就是被注释掉的，Colab 预装的 GPU 版 torch 直接用。）

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

Colab 的 `/content` 是**临时盘**，实例一释放全没。而续训需要的 optimizer state / scaler / epoch / step 都在 checkpoint 里，丢了只能从头。

⚠️ **特别注意**：`train_pretrain.py` 和 `train_full_sft.py` 里 checkpoint 路径是**硬编码**的：

```python
lm_checkpoint(..., save_dir='../checkpoints')   # --save_dir 管不到它！
```

`--save_dir` 只控制模型权重输出目录，管不了这个 resume 检查点。

✅ **最省事的解法：把整个项目放 Drive**，让这个硬编码路径天然落在持久存储上：

```bash
git clone --depth 1 https://github.com/Ezra-Maker-MAX/minimind.git /content/drive/MyDrive/minimind
```

这样就不用软链，也不用改源码。

---

## 二、目录规划

```
/content/drive/MyDrive/minimind/     ← 项目（持久）
├── checkpoints/    ← 断点续训状态：optimizer / scaler / epoch / step
├── out/            ← 模型权重 *.pth
├── logs/           ← train.log + last_cmd.sh（启动命令存档，保证续训参数一致）
└── dataset/        ← 数据集（持久，避免每次重下）

/content/minimind/dataset/           ← 每次会话的工作副本（VM 本地盘，训练实际读这里）
```

**为什么数据集要 copy 到本地盘**：Drive 走 FUSE，训练时 dataloader 反复读会被拖慢；VM 的 `/content` 是本地 NVMe。用 `shutil.copy2` 保留 mtime，还能让 `datasets` 复用 arrow 缓存，省掉每次 jsonl→arrow 的转换。

### 数据集体积（决定你能用哪套）

| 组合 | 文件 | 体积 | 免费 Drive 15GB |
|---|---|---|---|
| **mini（推荐）** | `pretrain_t2t_mini` + `sft_t2t_mini` | **2.8 GB** | ✅ |
| 完整版 | `pretrain_t2t` + `sft_t2t` | **24 GB** | ❌ 装不下 |

免费层只能用 mini 组合。RL 阶段数据（`dpo.jsonl` 53MB + `rlaif.jsonl` 24MB）无压力。

---

## 三、快速开始

```bash
# 0) Colab 菜单：修改 → 笔记本设置 → 硬件加速器 → T4 GPU → 保存

# 1) 装依赖（绝不要 -r requirements.txt）
!pip -q install "numpy>=2.1" datasets transformers einops rich huggingface_hub

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
    --num_workers 2 --dtype float16 \
    --data_path /content/minimind/dataset/pretrain_t2t_mini.jsonl \
    --log_interval 50 --save_interval 500 --from_resume 1

# 6) SFT
!python train_full_sft.py --epochs 1 --batch_size 8 --accumulation_steps 2 \
    --num_workers 2 --dtype float16 \
    --data_path /content/minimind/dataset/sft_t2t_mini.jsonl \
    --log_interval 50 --save_interval 500 --from_resume 1
```

参数说明：
- `--num_workers 2`：脚本默认 8，但免费层只有 2~4 核，开多了反而慢
- `--save_interval 500`：默认 1000，缩短以降低断点损失
- `--from_resume 1`：**必须带**，检测到 `checkpoints/` 就自动接上

---

## 四、断点续训

Colab 免费层会话随时可能被回收，所以**一切以"能接着跑"为前提设计**：

1. 产物全在 Drive（见上表），实例释放不丢
2. 训练命令统一带 `--from_resume 1`，重跑自动加载 `checkpoints/` 里的 optimizer / scaler / epoch / step
3. 启动命令存档到 `logs/last_cmd.sh`，保证续训时参数与上次完全一致
4. 日志 `logs/train.log` 也同步到 Drive，新会话能看到历史 loss

被中断后，只需重新运行**同一条命令**即可，`--from_resume 1` 会从断点继续。

⚠️ 注意：Colab **没有外部唤醒通道**，实例被回收后无法自动重启，这一步必须人工（或用浏览器自动化）触发。

---

## 五、时长预期

README 标称基于单卡 3090：pretrain_mini ≈1.21h/epoch、sft_mini ≈1.10h/epoch。
**T4 约为 3090 的 40~50% 算力**，换算下来：

| 阶段 | T4 预估（1 epoch） |
|---|---|
| pretrain（mini） | **2.5 ~ 3 小时** |
| SFT（mini） | **2.5 ~ 3 小时** |

合计约 6 小时起步，**远超免费层单次会话的稳定区间**。所以：
- `--epochs 1`（脚本默认 2）
- `--from_resume 1` + Drive 持久化
- 分多次会话跑完

**显存不是瓶颈**：64M 模型训练时参数+梯度+AdamW 状态约 1GB 出头，T4 的 16GB 绰绰有余。瓶颈是算力。

---

## 六、自动化脚本

仓库里的 `tools/colab_runner.py` 封装了全流程（握手、执行、长任务后台化、日志轮询、断点续训）：

```bash
python colab_runner.py prepare --data mini          # 一次性：装依赖+挂Drive+灌数据
python colab_runner.py pretrain --skip-prepare --watch 120   # 训练并轮询日志 120 分钟
python colab_runner.py sft      --skip-prepare --watch 120
python colab_runner.py status                       # 重连看日志，不启动新任务
python colab_runner.py smoke                        # 冒烟：200 行假数据跑通一个 step
```

设计要点：
- 训练用 `start_new_session=True` 后台起 + 日志落盘，避免单次调用阻塞几小时
- 每次看日志顺手把 `train.log` 备份到 Drive
- 启动前 `pgrep` 检查，避免重复起进程
- 唯一需要人工的一步：浏览器点「连接」

---

## 七、排障

| 现象 | 原因 / 处理 |
|---|---|
| 所有 cell 返回空、`execution_count` 为 null | kernel 被前序 cell 占死（常见于 pip 依赖回溯）→ Colab UI 点停止 |
| `CUDA out of memory` | 调小 `--batch_size`，或加 `--accumulation_steps` 补偿有效 batch |
| loss 不降 / NaN | T4 上忘改 dtype → 加 `--dtype float16` |
| 重跑后从头开始 | 没带 `--from_resume 1`，或 `checkpoints/` 不在 Drive 上 |
| 训练极慢 | 数据集在读 Drive 而不是本地盘 → copy 到 `/content` 再用 |
| `flash_attn` 相关报错 | 无需担心，项目用的是 PyTorch 内置 `F.scaled_dot_product_attention`，不用装 flash-attn |

---

## 八、还没验证的部分

- A100 会话下的 bfloat16 路径（免费层多数时候只给 T4）
- RL 阶段（DPO / PPO / GRPO）在 T4 上的可行性与耗时
- 完整版 24GB 数据集（受限于免费 Drive 15GB，未测）

---

*基于 2026-09 实测整理。Colab 政策与镜像环境变化较快，若流程跑不通请先看第七节。*
