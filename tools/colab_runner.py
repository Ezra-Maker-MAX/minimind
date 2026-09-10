# -*- coding: utf-8 -*-
"""
colab_runner.py — 通过本地 colab-mcp 驱动 Colab 自动跑流水线

解决的问题：
  1. 握手：自动触发浏览器打开带 token 的 Colab 页面并等待回连（可设超时）
  2. 执行：自动 add_code_cell + run_code_cell 并取回输出
  3. 卡死检测：任何一步超时立刻报错退出，不再干等（MCP 工具没有 interrupt）
  4. 长任务：训练用 start_new_session 后台起，日志落盘，脚本轮询 tail
  5. 断点续训：产物全放 Google Drive，脚本重连后接着看日志/接着跑

用法：
  python colab_runner.py smoke      # 环境体检 + 装依赖 + clone + 小样本跑通一步
  python colab_runner.py pretrain   # P1 后台启动预训练 + 持续轮询日志
  python colab_runner.py sft        # P1 后台启动 SFT + 持续轮询日志
  python colab_runner.py dpo        # P2 对齐（DPO，无需 reward model）
  python colab_runner.py lora --lora-name lora_medical   # P3 LoRA 定制
  python colab_runner.py status     # 只看训练日志和产出，不启动新任务
  python colab_runner.py exec --file step.py   # 执行自定义脚本

提速：任何训练命令追加 --use-compile 1 可开启 torch.compile
     （T4 算力仅 3090 的 40~50%，pretrain 两万+ step 收益明显；
       注意首次 step 有 2~5 分钟编译开销）

数据组合：mini=2.98GB(P1) / full=22.4GB(免费Drive装不下) / p2=178MB(对齐) / p3=59MB(LoRA)

约定：
  - 项目常驻 Google Drive：/content/drive/MyDrive/minimind
    （这样 train_pretrain.py 里硬编码的 ../checkpoints 天然落在 Drive，实例释放不丢）
  - 训练日志：/content/train.log（VM 本地盘，写得快；要长期留可自行 copy 到 Drive）
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

EXE = r"C:\Users\lenovo\.local\bin\colab-mcp.exe"
LOGDIR = r"C:\Users\lenovo\.workbuddy\logs\colab-mcp"

REPO = "https://github.com/Ezra-Maker-MAX/minimind.git"
PROJ = "/content/drive/MyDrive/minimind"
LOG = "/content/train.log"


# ────────────────────────────── MCP stdio 客户端 ──────────────────────────────
class McpClient:
    """极简 MCP stdio JSON-RPC 客户端，够用就好。"""

    def __init__(self, exe=EXE, handshake_timeout=150, logdir=LOGDIR):
        if not os.path.exists(exe):
            raise SystemExit(f"找不到 colab-mcp: {exe}")
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        self.p = subprocess.Popen(
            [exe, "--log", logdir], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=0,
        )
        self.q = queue.Queue()
        self._id = 0
        threading.Thread(target=self._reader, daemon=True).start()
        self._rpc("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "colab-runner", "version": "1.0"},
        }, timeout=60)
        self._send("notifications/initialized", None, notify=True)

    def _reader(self):
        for raw in self.p.stdout:
            raw = raw.strip()
            if not raw:
                continue
            try:
                self.q.put(json.loads(raw))
            except Exception:
                pass

    def _send(self, method, params=None, notify=False):
        if notify:
            msg = {"jsonrpc": "2.0", "method": method}
            if params is not None:
                msg["params"] = params
        else:
            self._id += 1
            msg = {"jsonrpc": "2.0", "id": self._id, "method": method,
                   "params": params or {}}
        self.p.stdin.write((json.dumps(msg) + "\n").encode())
        self.p.stdin.flush()
        return None if notify else self._id

    def _rpc(self, method, params=None, timeout=120):
        mid = self._send(method, params)
        return self._wait(mid, timeout)

    def _wait(self, mid, timeout):
        deadline = time.time() + timeout
        while True:
            left = deadline - time.time()
            if left <= 0:
                raise TimeoutError(f"等待响应超时({timeout}s)，kernel 可能正忙")
            try:
                msg = self.q.get(timeout=min(1.0, left))
            except queue.Empty:
                continue
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"MCP error: {msg['error']}")
                return msg.get("result", {})

    def call(self, name, args=None, timeout=180):
        """调用 MCP 工具，返回解析后的 Python 对象。"""
        res = self._rpc("tools/call",
                        {"name": name, "arguments": args or {}}, timeout)
        if res.get("isError"):
            raise RuntimeError(f"工具 {name} 报错: {res}")
        txt = ""
        for c in res.get("content", []):
            if c.get("type") == "text":
                txt += c.get("text", "")
        if not txt:
            return {}
        try:
            return json.loads(txt)
        except Exception:
            return {"_text": txt}

    def wait_token(self, timeout=30):
        """从日志里抓一次性握手密钥。
        依赖对 colab_mcp/websocket_server.py 打的补丁行：
            logging.info(f"MCP_PROXY_TOKEN={t} MCP_PROXY_PORT={p}")
        """
        import glob
        import re
        deadline = time.time() + timeout
        seen = set()
        while time.time() < deadline:
            for f in sorted(glob.glob(os.path.join(self.logdir, "*.log")),
                            key=os.path.getmtime, reverse=True)[:5]:
                try:
                    txt = open(f, encoding="utf-8", errors="ignore").read()
                except Exception:
                    continue
                for m in re.finditer(
                        r"MCP_PROXY_TOKEN=(\S+)\s+MCP_PROXY_PORT=(\d+)", txt):
                    if m.group(0) not in seen:
                        seen.add(m.group(0))
                        return m.group(1), int(m.group(2))
            time.sleep(1)
        return None, None

    def close(self):
        try:
            self.p.terminate()
        except Exception:
            pass


# ────────────────────────────── Colab 操作封装 ──────────────────────────────
class Colab:
    def __init__(self, handshake_timeout=150):
        self.m = McpClient()
        self.handshake_timeout = handshake_timeout

    # -- 握手 ---------------------------------------------------------------
    def connect(self):
        token, port = self.m.wait_token(30)
        url = None
        if token:
            url = ("https://colab.research.google.com/notebooks/empty.ipynb"
                   f"#mcpProxyToken={token}&mcpProxyPort={port}")
            print("\n[握手] 手动备用链接（自动弹窗失败时复制到浏览器打开）：")
            print("   " + url + "\n")
        else:
            print("[握手] 未能从日志读到 token —— 若 handshake 失败，"
                  "检查 colab_mcp/websocket_server.py 里的 token 日志补丁是否还在。")
        print("[握手] 正在打开浏览器…若没弹出，请手动打开上面的链接。")
        print("[握手] 页面加载后点「连接」分配运行时（GPU 需先在 修改→笔记本设置 选好）。")
        t0 = time.time()
        try:
            r = self.m.call("open_colab_browser_connection", {},
                            timeout=self.handshake_timeout)
        except TimeoutError:
            raise SystemExit(
                f"[握手] 超时 {self.handshake_timeout}s 未连上。\n"
                "  常见原因：① 没点页面上的「连接」；② 弹窗被浏览器拦截；\n"
                "  ③ 已有另一个 colab-mcp 实例独占（返回 1013）。"
            )
        # 返回值可能是裸 bool（json "true"）或 dict，都要兼容
        if isinstance(r, bool):
            ok = r
        elif isinstance(r, dict):
            ok = (r.get("result") is True
                  or str(r.get("_text", "")).strip().lower() == "true")
        else:
            ok = str(r).strip().lower() == "true"
        if not ok:
            raise SystemExit(f"[握手] 失败，返回: {r!r}")
        print(f"[握手] 成功（{time.time() - t0:.0f}s）")

    # -- 执行 ---------------------------------------------------------------
    def exec_code(self, code, timeout=300, show=True, tag=""):
        """插入一个 code cell 并执行，返回 stdout 文本。"""
        r = self.m.call("add_code_cell",
                        {"cellIndex": 9999, "language": "python", "code": code},
                        timeout=60)
        cid = r.get("newCellId")
        if not cid:
            raise RuntimeError(f"创建 cell 失败: {r}")
        out = self.m.call("run_code_cell", {"cellId": cid}, timeout=timeout)
        text = self._flatten(out)
        if show:
            head = f"── {tag or 'output'} " + "─" * 40
            print(head)
            print(text if text.strip() else "(无输出 —— 注意：execution_count 为空说明 cell 在排队/未执行)")
            print("─" * len(head))
        return text

    @staticmethod
    def _flatten(out):
        parts = []
        for o in (out.get("outputs") or []):
            if o.get("output_type") == "stream":
                parts.append("".join(o.get("text", [])))
            elif o.get("output_type") == "error":
                parts.append("ERROR: " + "\n".join(o.get("traceback", [])))
            elif o.get("output_type") == "execute_result":
                parts.append(str(o.get("data", {}).get("text/plain", "")))
        if not parts and out.get("_text"):
            parts.append(out["_text"])
        return "".join(parts)


# ────────────────────────────── 流水线步骤 ──────────────────────────────
ENV_CHECK = """
import sys, torch
print('Python', sys.version.split()[0])
print('torch ', torch.__version__)
if not torch.cuda.is_available():
    raise SystemExit('NO_GPU: 请 修改→笔记本设置→硬件加速器→T4 GPU，然后重新运行')
p = torch.cuda.get_device_properties(0)
print('GPU   :', p.name, '| VRAM', round(p.total_memory/1e9,1), 'GB | sm_%d%d' % (p.major, p.minor))
print('bf16 hw:', p.major >= 8)
globals()['DTYPE'] = 'bfloat16' if p.major >= 8 else 'float16'
print('>>> dtype =', DTYPE)
"""

INSTALL = r"""
# 只装训练必需项；绝不能 pip install -r requirements.txt（numpy==1.26.4 在 py3.13 无 wheel）
# 清单由 requirements.txt + 逐个 grep 训练脚本 import 核对得出，比"只装 6 个包"稳妥
!pip -q install "numpy>=2.1" datasets transformers einops rich huggingface_hub \
    jsonlines datasketch simhash psutil jieba nltk scikit_learn trl wandb swanlab
import numpy, datasets, transformers
print('numpy', numpy.__version__, '| datasets', datasets.__version__, '| transformers', transformers.__version__)
# 逐个验证训练脚本真正 import 的包，避免跑到一半才崩
for _m in ['rich', 'jsonlines', 'psutil', 'einops', 'wandb', 'trl']:
    try:
        __import__(_m); print('  ok  ', _m)
    except Exception as _e:
        print('  MISS', _m, _e)
"""

MOUNT_AND_CLONE = r"""
from google.colab import drive
drive.mount('/content/drive')
import os
!mkdir -p /content/drive/MyDrive/minimind
%cd /content/drive/MyDrive/minimind
if not os.path.exists('/content/drive/MyDrive/minimind/trainer'):
    !git clone -q --depth 1 REPO_URL /content/drive/MyDrive/minimind_tmp
    !cp -r /content/drive/MyDrive/minimind_tmp/. /content/drive/MyDrive/minimind/
    !rm -rf /content/drive/MyDrive/minimind_tmp
print('项目就绪:', os.getcwd())
!ls
"""

SMOKE = r"""
import sys, os, json
sys.path.insert(0, PROJ)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoTokenizer
cfg = MiniMindConfig()
model = MiniMindForCausalLM(cfg)
print('参数量: %.1fM' % (sum(p.numel() for p in model.parameters())/1e6))
tok = AutoTokenizer.from_pretrained(PROJ + '/model')
print('tokenizer 词表:', len(tok))
os.makedirs(PROJ + '/dataset', exist_ok=True)
with open(PROJ + '/dataset/smoke.jsonl','w',encoding='utf-8') as f:
    for i in range(200):
        f.write(json.dumps({'text':'今天天气很好，我们去公园散步。第%d条测试数据。'%i}, ensure_ascii=False)+'\n')
print('smoke.jsonl OK')
"""

SMOKE_TRAIN = r"""
import subprocess, os, sys, time
cmd = ('cd PROJ/trainer && python train_pretrain.py --epochs 1 --batch_size 8 '
       '--accumulation_steps 1 --num_workers 1 --dtype DTYPE '
       '--data_path ../dataset/smoke.jsonl --log_interval 5 --save_interval 1000')
p = subprocess.Popen(['bash','-lc',cmd], stdout=open('/content/smoke.log','w'),
                     stderr=subprocess.STDOUT, start_new_session=True)
for _ in range(60):
    time.sleep(2)
    if p.poll() is not None: break
print('exit:', p.poll())
print(open('/content/smoke.log').read()[-3000:])
"""

# 数据集体积（2026-09 实测真实大小，非 README 标称）
#   mini 组合 2.98GB；完整版 22.4GB（免费 Drive 15GB 装不下）；P2+P3 合计仅 236MB
DATA_SETS = {
    "mini": ["pretrain_t2t_mini.jsonl", "sft_t2t_mini.jsonl"],            # 2.98GB  P1 基座
    "full": ["pretrain_t2t.jsonl", "sft_t2t.jsonl"],                      # 22.4GB  装不下
    "p2":   ["dpo.jsonl", "rlaif.jsonl", "agent_rl.jsonl",
             "agent_rl_math.jsonl"],                                      # 178MB   P2 对齐
    "p3":   ["lora_medical.jsonl", "lora_identity.jsonl",
             "lora_exam.jsonl"],                                          # 59MB    P3 定制
    "rl":   ["dpo.jsonl", "rlaif.jsonl"],                                 # 77MB    兼容旧名
}
# 各组合预估占用（GB），用于 Drive 空间检查
DATA_SIZE = {"mini": 2.98, "full": 22.4, "p2": 0.18, "p3": 0.06, "rl": 0.08}
LOCAL_DATA = "/content/minimind/dataset"   # VM 本地盘：训练实际读取处，速度快

PREPARE_DATA = r"""
# 一次性：把数据集灌进 Google Drive，之后所有会话复用
import os, shutil
from huggingface_hub import hf_hub_download

DD = PROJ + '/dataset'
os.makedirs(DD, exist_ok=True)

def free_gb(p):
    s = os.statvfs(p)
    return s.f_bavail * s.f_frsize / 1e9

print('Drive 剩余空间: %.1f GB' % free_gb('/content/drive'))
print('本地盘剩余空间: %.1f GB' % free_gb('/content'))

files = FILES_PLACEHOLDER
need = DATA_SIZE_PLACEHOLDER
print('本次需要约 %.2f GB' % need)
if free_gb('/content/drive') < need + 1:
    raise SystemExit('Drive 空间不足！完整版 22.4GB 超过免费 Drive 15GB，'
                     '请改用 mini / p2 / p3 组合，或清理 Drive / 升级存储')

for fn in files:
    dst = os.path.join(DD, fn)
    if os.path.exists(dst) and os.path.getsize(dst) > 1024:
        print('[跳过] 已在 Drive:', fn, round(os.path.getsize(dst)/1e6, 1), 'MB')
        continue
    t0 = __import__('time').time()
    p = hf_hub_download(repo_id='jingyaogong/minimind_dataset', filename=fn,
                        repo_type='dataset', local_dir=DD)
    dt = __import__('time').time() - t0
    sz = os.path.getsize(p) / 1e6
    print('[下载] %s %.1f MB 用时 %.0fs (%.1f MB/s)' % (fn, sz, dt, sz/max(dt,1)))
print('数据集就绪于 Drive:', DD)
"""

SYNC_DATA = r"""
# 每次会话：Drive → VM 本地盘。别在训练时直接读 Drive，FUSE 会拖慢 dataloader
import os, shutil, time
DD = PROJ + '/dataset'
LD = 'LOCAL_DATA_PLACEHOLDER'
os.makedirs(LD, exist_ok=True)
for fn in FILES_PLACEHOLDER:
    src, dst = os.path.join(DD, fn), os.path.join(LD, fn)
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        print('[本地已有]', fn, round(os.path.getsize(dst)/1e6, 1), 'MB')
        continue
    t0 = time.time()
    shutil.copy2(src, dst)          # copy2 保留 mtime，便于 datasets 复用 arrow 缓存
    dt = time.time() - t0
    sz = os.path.getsize(dst) / 1e6
    print('[同步] %s %.1f MB 用时 %.0fs (%.0f MB/s)' % (fn, sz, dt, sz/max(dt,1)))
"""


def train_launch(script, data, batch, accum, epochs=1, use_compile=0, extra=""):
    """生成后台启动训练的 cell 代码（nohup 式，配合日志轮询）。"""
    comp = " --use_compile 1" if use_compile else ""
    return f"""
import subprocess, os, time
proj = '{PROJ}'
log = '{LOG}'
# 若已有训练在跑，不重复启动
alive = os.popen("pgrep -f '{script}' | tr '\\n' ' '").read().strip()
if alive:
    print('已有进程在跑, pid:', alive)
else:
    os.system(f'cp {{log}} {{log}}.prev 2>/dev/null')
    cmd = (f'cd {{proj}}/trainer && python {script} --epochs {epochs} '
           f'--batch_size {batch} --accumulation_steps {accum} --num_workers 2 '
           f'--dtype DTYPE_PLACEHOLDER --data_path {LOCAL_DATA}/{data} '
           f'--log_interval 50 --save_interval 500 --from_resume 1{comp}EXTRA_PLACEHOLDER')
    # 把启动命令存 Drive：新会话可直接复用，保证 --from_resume 参数一致
    os.makedirs(proj + '/logs', exist_ok=True)
    open(proj + '/logs/last_cmd.sh','w').write('#!/bin/bash\\n' + cmd + '\\n')
    subprocess.Popen(['bash','-lc', cmd], stdout=open(log,'w'),
                     stderr=subprocess.STDOUT, start_new_session=True)
    print('已后台启动:', cmd)
    if {use_compile}:
        print('NOTE: 已开启 --use_compile，首次 step 需 2~5 分钟编译，属正常')
time.sleep(20)
print(open(log).read()[-2000:] if os.path.exists(log) else '(日志还没生成)')
""".replace("EXTRA_PLACEHOLDER", (" " + extra if extra else ""))

TAIL = r"""
import os, shutil
log = 'LOGPATH'
if not os.path.exists(log):
    print('(无日志)')
else:
    n = os.path.getsize(log)
    with open(log) as f:
        f.seek(max(0, n - 4000))
        print(f.read())
    # 每次看日志顺手备份到 Drive：断点后新会话也能看到历史 loss 曲线
    try:
        lg = PROJ + '/logs'
        os.makedirs(lg, exist_ok=True)
        shutil.copy(log, lg + '/train.log')
    except Exception as e:
        print('[日志备份失败]', e)
print('--- 进程 ---')
print(os.popen("pgrep -af 'train_' | head -5").read() or '(无训练进程)')
print('--- Drive 上的产物 ---')
for d in ['checkpoints', 'out', 'logs']:
    p = os.path.join(PROJ, d)
    if os.path.isdir(p):
        fs = sorted(os.listdir(p))[-6:]
        print(' %-12s %s' % (d + '/', fs))
"""


# ────────────────────────────── 命令实现 ──────────────────────────────
def _fill(code, data_set="mini"):
    files = repr(DATA_SETS.get(data_set, DATA_SETS["mini"]))
    need = DATA_SIZE.get(data_set, 2.98)
    return (code.replace("REPO_URL", REPO)
                .replace("LOCAL_DATA_PLACEHOLDER", LOCAL_DATA)
                .replace("FILES_PLACEHOLDER", files)
                .replace("DATA_SIZE_PLACEHOLDER", repr(need))
                .replace("SET_PLACEHOLDER", data_set)
                .replace("PROJ", PROJ)
                .replace("LOGPATH", LOG))


def cmd_smoke(args):
    c = Colab(args.handshake)
    c.connect()
    c.exec_code(ENV_CHECK, timeout=120, tag="环境体检")
    c.exec_code(_fill(INSTALL), timeout=900, tag="装依赖")
    c.exec_code(_fill(MOUNT_AND_CLONE), timeout=600, tag="挂 Drive + clone")
    c.exec_code(_fill(SMOKE), timeout=300, tag="模型实例化")
    c.exec_code(_fill(SMOKE_TRAIN), timeout=600, tag="冒烟训练")
    print("\n冒烟完成。若上面出现 loss 数值，说明全链路打通。")


def _dtype_cell(c):
    """先跑一次环境体检，把 DTYPE 写进 notebook 全局，供后续 cell 使用。"""
    c.exec_code(ENV_CHECK.replace(
        "globals()['DTYPE'] =",
        "globals()['DTYPE'] ="), timeout=120, tag="环境体检(确定 dtype)")


def cmd_train(args):
    c = Colab(args.handshake)
    c.connect()
    c.exec_code(_fill(INSTALL), timeout=900, tag="装依赖")
    c.exec_code(_fill(MOUNT_AND_CLONE), timeout=600, tag="挂 Drive + clone")
    c.exec_code(ENV_CHECK, timeout=120, tag="环境体检")

    stage = args.stage
    extra = ""
    if stage == "pretrain":
        script, data, bs, ac = "train_pretrain.py", DATA_SETS[args.data][0], 16, 8
    elif stage == "sft":
        script, data, bs, ac = "train_full_sft.py", DATA_SETS[args.data][1], 8, 2
    elif stage == "dpo":
        # DPO 不需要 reward model，直接可跑
        script, data, bs, ac = "train_dpo.py", "dpo.jsonl", 4, 2
        extra = "--from_weight full_sft"
    elif stage == "lora":
        script, data, bs, ac = "train_lora.py", f"{args.lora_name}.jsonl", 16, 1
        extra = f"--lora_name {args.lora_name} --from_weight full_sft"
    else:
        raise SystemExit(f"未知阶段: {stage}")

    if not args.skip_prepare:
        c.exec_code(_fill(PREPARE_DATA, args.data), timeout=3600,
                    tag=f"数据集({args.data})灌入 Drive")
    c.exec_code(_fill(SYNC_DATA, args.data), timeout=1800, tag="同步到 VM 本地盘")
    code = train_launch(script, data, bs, ac, args.epochs,
                        use_compile=args.use_compile, extra=extra)
    # dtype 由环境体检写入全局 DTYPE；这里用 python 变量拼接，避免硬编码
    code = code.replace("DTYPE_PLACEHOLDER", "' + DTYPE + '")
    c.exec_code(_fill(code, args.data), timeout=600, tag=f"启动 {stage}")
    if args.watch:
        poll(c, args.watch)


def cmd_prepare(args):
    """只灌数据，不训练。适合先单独跑一次把 Drive 填满。"""
    c = Colab(args.handshake)
    c.connect()
    c.exec_code(_fill(INSTALL), timeout=900, tag="装依赖")
    c.exec_code(_fill(MOUNT_AND_CLONE), timeout=600, tag="挂 Drive + clone")
    c.exec_code(_fill(PREPARE_DATA, args.data), timeout=5400,
                tag=f"数据集({args.data})灌入 Drive")
    c.exec_code(_fill(SYNC_DATA, args.data), timeout=1800, tag="同步到 VM 本地盘")
    print("\n数据已就绪。之后跑训练时加 --skip-prepare 就不会重复下载。")


def cmd_status(args):
    c = Colab(args.handshake)
    c.connect()
    for i in range(max(1, args.watch)):
        c.exec_code(_fill(TAIL), timeout=120, tag="训练日志")
        if i < args.watch - 1:
            time.sleep(60)


def cmd_exec(args):
    code = open(args.file, encoding="utf-8").read()
    c = Colab(args.handshake)
    c.connect()
    c.exec_code(code, timeout=args.timeout, tag=os.path.basename(args.file))


def poll(c, minutes):
    print(f"\n开始轮询日志 {minutes} 分钟（Ctrl+C 可随时退出，Colab 里的训练不会停）")
    end = time.time() + minutes * 60
    while time.time() < end:
        time.sleep(60)
        try:
            c.exec_code(_fill(TAIL), timeout=120, tag=f"日志 {time.strftime('%H:%M:%S')}")
        except (TimeoutError, RuntimeError) as e:
            print(f"[轮询] 本次读取失败({e})，继续…")
    print("\n轮询结束。训练仍在 Colab 上跑，下次用 `status` 继续看。")


def main():
    ap = argparse.ArgumentParser(description="colab_runner — 自动驱动 Colab 跑流水线")
    ap.add_argument("cmd", choices=["smoke", "prepare", "pretrain", "sft", "dpo", "lora",
                                    "status", "exec"])
    ap.add_argument("--handshake", type=int, default=150, help="握手等待秒数(默认150)")
    ap.add_argument("--watch", type=int, default=0, help="启动后轮询日志的分钟数")
    ap.add_argument("--epochs", type=int, default=1, help="训练 epoch 数")
    ap.add_argument("--data", choices=["mini", "full", "p2", "p3", "rl"], default="mini",
                    help="数据组合：mini=2.98GB(P1) / full=22.4GB(装不下) / "
                         "p2=178MB(对齐) / p3=59MB(LoRA) / rl=77MB(兼容)")
    ap.add_argument("--use-compile", type=int, default=0, choices=[0, 1],
                    help="开启 torch.compile 提速（首次 step 需 2~5 分钟编译）")
    ap.add_argument("--lora-name", default="lora_medical",
                    help="LoRA 权重名：lora_medical / lora_identity")
    ap.add_argument("--skip-data", action="store_true", help="跳过数据集下载")
    ap.add_argument("--skip-prepare", action="store_true",
                    help="跳过数据下载(Drive 里已有)，只做 Drive→本地盘同步")
    ap.add_argument("--file", help="exec 模式下的脚本文件")
    ap.add_argument("--timeout", type=int, default=600, help="exec 模式超时秒")
    a = ap.parse_args()

    if a.cmd == "smoke":
        cmd_smoke(a)
    elif a.cmd == "prepare":
        cmd_prepare(a)
    elif a.cmd in ("pretrain", "sft", "dpo", "lora"):
        a.stage = a.cmd
        if a.cmd in ("dpo", "lora") and a.data == "mini":
            a.data = "p2" if a.cmd == "dpo" else "p3"   # 自动纠正默认数据集
        cmd_train(a)
    elif a.cmd == "status":
        cmd_status(a)
    else:
        if not a.file:
            raise SystemExit("exec 需要 --file")
        cmd_exec(a)


if __name__ == "__main__":
    main()
