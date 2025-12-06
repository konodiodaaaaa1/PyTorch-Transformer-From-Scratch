import torch
import torch.nn as nn
import torch.amp as amp
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from tokenizers import Tokenizer
from pathlib import Path
import sys
import os
import math

from train_utils import create_padding_mask, create_look_ahead_mask

# --- 确保能找到我们的模块 ---
sys.path.append(str(Path(__file__).parent))
try:
    from model import Transformer
except ImportError as e:
    print(f"错误: 找不到 model.py。 {e}")
    sys.exit(1)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----------------------------------------------------------------------
# 【阶段二 & 五】：数据管道 (已修正所有 Bug)
# ----------------------------------------------------------------------

# --- 1. 加载分词器和全局变量 ---
print("正在加载分词器...")
SCRIPT_DIR = Path(__file__).parent
SRC_TOKENIZER_PATH = SCRIPT_DIR / "src_tokenizer.json"
TGT_TOKENIZER_PATH = SCRIPT_DIR / "tgt_tokenizer.json"

src_tokenizer = Tokenizer.from_file(str(SRC_TOKENIZER_PATH))
tgt_tokenizer = Tokenizer.from_file(str(TGT_TOKENIZER_PATH))

PAD_IDX_SRC = src_tokenizer.token_to_id("<pad>")
PAD_IDX_TGT = tgt_tokenizer.token_to_id("<pad>")
SOS_IDX_TGT = tgt_tokenizer.token_to_id("<sos>")
EOS_IDX_TGT = tgt_tokenizer.token_to_id("<eos>")

assert PAD_IDX_SRC == PAD_IDX_TGT, "Pad ID 必须相同 (通常为 0)"
SRC_VOCAB_SIZE = src_tokenizer.get_vocab_size()
TGT_VOCAB_SIZE = tgt_tokenizer.get_vocab_size()


# --- 2. (关键修正) Collate (整理) 函数 ---
class TranslationDataset(Dataset):
    def __init__(self, data, src_tokenizer, tgt_tokenizer):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        src_text = sample['classical']
        tgt_text = sample['modern']

        # --- (这是 Bug #3/4/5 的核心修复) ---

        # 1. 源 (文言文): 只需要 ID
        src_ids = self.src_tokenizer.encode(src_text).ids

        # 2. 目标 (现代文): 只需要 ID
        tgt_ids = self.tgt_tokenizer.encode(tgt_text).ids

        # 3. 创建 "输入" 和 "标签"
        #    注意: 我们在这里 *只* 添加 SOS / EOS
        #    pad_sequence 会在 collate_fn 中处理 <pad>

        # 解码器输入: [SOS] + sentence
        tgt_input = torch.tensor([SOS_IDX_TGT] + tgt_ids, dtype=torch.long)

        # 标签: sentence + [EOS]
        tgt_label = torch.tensor(tgt_ids + [EOS_IDX_TGT], dtype=torch.long)

        # 源输入:
        src_input = torch.tensor(src_ids, dtype=torch.long)

        # 返回一个字典，这才是标准做法
        return {
            "src_input": src_input,
            "tgt_input": tgt_input,
            "tgt_label": tgt_label
        }


# 2. (新) 重写 setup_dataloaders 来使用这个 Dataset
def setup_dataloaders(
        train_data,  # (这是 HuggingFace Dataset 对象, e.g., dataset['train'])
        val_data,  # (这是 HuggingFace Dataset 对象, e.g., dataset['validation'])
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        batch_size: int,
        max_allowed_len: int  # (我们暂时不用这个，但保留接口)
):
    print("  正在构建 Torch Dataset...")

    # --- (过滤掉过长的句子) ---
    # (注意: 我们在 tokenizer 之后做这个过滤可能更准,
    #  但基于字符的过滤已经足够好了，而且快得多)

    def filter_long(example):
        return len(example['classical']) < max_allowed_len and \
            len(example['modern']) < max_allowed_len

    print(f"  过滤前 (Train): {len(train_data)} / (Val): {len(val_data)}")

    train_data = train_data.filter(filter_long)
    val_data = val_data.filter(filter_long)

    print(f"  过滤后 (Train): {len(train_data)} / (Val): {len(val_data)}")

    # --- (实例化我们自己的 Dataset) ---
    train_dataset_torch = TranslationDataset(train_data, src_tokenizer, tgt_tokenizer)
    val_dataset_torch = TranslationDataset(val_data, src_tokenizer, tgt_tokenizer)

    print("  创建 Collate Function...")

    # --- (!!!) 这才是正确的 Collate Function (!!!) ---
    # 它现在的工作 *非常* 简单：
    # 1. 从 batch (列表) 中取出 'src_input' 张量，放进一个新列表
    # 2. 把这个列表 pad 成一个大张量
    # 3. 对 'tgt_input' 和 'tgt_label' 做同样的事

    def collate_fn(batch: list):
        # batch 是一个列表, 每一项是 __getitem__ 返回的字典
        # e.g., [ {'src_input': T1, 'tgt_input': T2, 'tgt_label': T3}, ... ]

        # (我们之前修复 Bug #3/4 的地方)
        src_batch = pad_sequence(
            [item['src_input'] for item in batch],
            batch_first=True, padding_value=PAD_IDX_SRC
        )

        tgt_inputs = pad_sequence(
            [item['tgt_input'] for item in batch],
            batch_first=True, padding_value=PAD_IDX_TGT
        )

        tgt_labels = pad_sequence(
            [item['tgt_label'] for item in batch],
            batch_first=True, padding_value=PAD_IDX_TGT
        )

        # (!!!) 关键: 返回 3 个张量
        return src_batch, tgt_inputs, tgt_labels

    print("  创建 DataLoader...")
    train_loader = DataLoader(
        train_dataset_torch,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # (你可以根据你的 CPU 调整)
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset_torch,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader


# ----------------------------------------------------------------------
# 【阶段五：系统运行】
# ----------------------------------------------------------------------

# --- 4. (关键修正) 初始化 (带断点续训) ---
def initialize_model(device: torch.device, checkpoint_path: str = None):
    D_MODEL = 512
    NUM_LAYERS = 6
    NUM_HEADS = 8
    D_FF = 2048
    MAX_LEN = 5000
    DROPOUT = 0.1
    LEARNING_RATE = 1e-4

    model = Transformer(
        SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL,
        NUM_LAYERS, NUM_HEADS, D_FF, MAX_LEN, DROPOUT
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX_TGT, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    warmup_steps = 4000
    d_model_inv_sqrt = D_MODEL ** (-0.5)

    def lr_scheduler_lambda(step_num):
        step_num += 1  # (从 1 开始计数)
        arg1 = step_num ** (-0.5)
        arg2 = step_num * (warmup_steps ** (-1.5))
        lr_scale_factor = 0.8
        return d_model_inv_sqrt * min(arg1, arg2) * lr_scale_factor

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler_lambda)

    start_epoch = 1
    best_val_loss = float('inf')

    # --- (关键) 加载检查点 (Checkpointing) ---
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"--- 正在从 {checkpoint_path} 加载检查点... ---")

        # --- (Bug 1 修复: 使用 device 对象加载) ---
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if 'scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"--- 已加载。将从 Epoch {start_epoch} 开始训练。---")
    else:
        print(f"--- 未找到检查点。正在从头开始训练... ---")
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # ----------------------------------------------------
        # (!!!) 关键修复: 确保 LayerNorm 参数初始化正确 (!!!)
        # ----------------------------------------------------
    for name, p in model.named_parameters():
        if 'weight' in name and p.dim() > 1:
            # 权重（W, Q, K, V, O）: Xavier 初始化
            nn.init.xavier_uniform_(p)
        elif 'bias' in name:
            # 偏置（b）: 初始化为 0
            p.data.fill_(0.0)
        elif 'norm' in name and 'weight' in name:
            # (!!! LayerNorm 权重/gamma): 初始化为 1 (!!!)
            p.data.fill_(1.0)
        elif 'norm' in name and 'bias' in name:
            # (!!! LayerNorm 偏置/beta): 初始化为 0 (!!!)
            p.data.fill_(0.0)

    print(f"模型已在 {device} 上初始化")
    return model, loss_fn, optimizer, start_epoch, best_val_loss, lr_scheduler


# --- 5. 验证函数 (evaluate) ---
def evaluate(model, dataloader, loss_fn, device, pad_idx_src, pad_idx_tgt):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for (src_data, tgt_input_data, tgt_label_data) in dataloader:
            src_batch = src_data.to(device)
            tgt_input = tgt_input_data.to(device)
            tgt_label = tgt_label_data.to(device)
            src_mask = create_padding_mask(src_batch, pad_idx_src).to(device)
            tgt_mask = create_look_ahead_mask(tgt_input, pad_idx_tgt).to(device)

            # --- (Bug 1 修复: 使用 device.type 并检查) ---
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(src_batch, tgt_input, src_mask, tgt_mask)
                loss = loss_fn(
                    logits.reshape(-1, logits.shape[-1]),
                    tgt_label.reshape(-1)
                )
            total_loss += loss.item()

    return total_loss / len(dataloader)


# --- 6. 训练循环 (train_one_epoch) ---
def train_one_epoch(model, train_loader, val_loader, loss_fn, optimizer,
                    device, scaler, pad_idx_src, pad_idx_tgt, epoch, lr_scheduler,
                    log_interval, validate_interval, best_val_loss):
    model.train()
    total_loss = 0
    data_iter = iter(train_loader)
    num_batches = len(train_loader)

    for batch_count in range(num_batches):

        try:
            src_data, tgt_input_data, tgt_label_data = next(data_iter)
        except StopIteration:
            break

        src_batch = src_data.to(device)
        tgt_input = tgt_input_data.to(device)
        tgt_label = tgt_label_data.to(device)

        src_mask = create_padding_mask(src_batch, pad_idx_src)
        tgt_mask = create_look_ahead_mask(tgt_input, pad_idx_tgt)

        # --- (Bug 1 修复: 使用 device.type 并检查) ---
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            logits = model(src_batch, tgt_input, src_mask, tgt_mask)
            loss = loss_fn(
                logits.reshape(-1, logits.shape[-1]),
                tgt_label.reshape(-1)
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if lr_scheduler is not None:
            lr_scheduler.step()

        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()

        if (batch_count + 1) % log_interval == 0:
            current_lr = lr_scheduler.get_last_lr()[0]
            print(
                f"  [Train] Epoch {epoch}, Batch {batch_count + 1}/{num_batches}, Loss: {loss.item():.4e},lr:{current_lr:.6e}")

        # (你要求的: 每 N 批次验证一次)
        if (batch_count + 1) % validate_interval == 0:
            print(f"  (--- 正在运行验证 @ Batch {batch_count + 1} ---)")
            val_loss = evaluate(model, val_loader, loss_fn, device, pad_idx_src, pad_idx_tgt)
            model.train()  # (切回训练模式)

            print(f"  [Validate] Epoch {epoch}, Batch {batch_count + 1}, Val Loss: {val_loss:.4e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"    (新低点! Val Loss 降至 {val_loss:.4e}。正在保存 'transformer_checkpoint.pth'...)")
                torch.save(model.state_dict(), "transformer_checkpoint.pth")

    return total_loss / num_batches, best_val_loss


# ----------------------------------------------------------------------
# 【主程序：启动器】
# ----------------------------------------------------------------------

if __name__ == '__main__':

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 正在使用设备: {DEVICE} ---")

    # --- 1. 设置超参数 ---
    BATCH_SIZE = 8
    NUM_EPOCHS = 1000
    MAX_SEQ_LEN = 100  # (过滤长度)
    LOG_INTERVAL = 100
    VALIDATE_INTERVAL = 500
    CHECKPOINT_FILE = "transformer_checkpoint.pth"

    # print("\n--- [阶段五] 准备数据加载器 (过拟合测试模式) ---")

    # 1. 加载完整数据集
    try:
        dataset = load_dataset('xmj2002/Chinese_modern_classical')
        print(f"  成功加载 HuggingFace 'xmj2002' 数据集。")
    except Exception as e:
        print(f"!! 错误: 无法加载 HuggingFace 数据集: {e}")
        sys.exit(1)

    # --- (!!!) 过拟合测试配置 (!!!) ---
    # print("=" * 60)
    # print("!!! 警告: 正在执行 [过拟合测试] 模式 !!!")

    # 2. (正确地) 从 'train' 子集中抽数据
    # try:
    #     overfit_data = dataset['train'].select(range(20))
    #     print(f"  (已成功抽取 {len(overfit_data)} 条数据用于“背诵”)")
    #
    # except Exception as e:
    #     print(f"!! 错误: 抽取数据失败: {e}")
    #     sys.exit(1)

    # dataset = overfit_data
    # 3. (新功能) 显示第 1 个样本
    # print("-" * 60)
    # print("  以下是我们将用于“背诵”的第 1 个样本：")
    # sample = dataset[0]
    # print(f"  [文言文]: {sample['classical']}")
    # print(f"  [现代文]: {sample['modern']}")
    # print("=" * 60)

    split_dataset_dict = dataset['train'].train_test_split(
        test_size=0.01,  # (1% 作为验证集)
        shuffle=True,
        seed=42  # (确保分割可复现)
    )

    # 4. 分配新的训练集和验证集
    #    (注意: .train_test_split() 会创建 'train' 和 'test' 两个新键)
    train_data_raw = split_dataset_dict['train']
    val_data_raw = split_dataset_dict['test']  # <--- (这是我们新分出来的验证集)

    print(f"  已成功分割: {len(train_data_raw)} 条训练数据 (99%)")
    print(f"  已成功分割: {len(val_data_raw)} 条验证数据 (1%)")

    # --- 2. 准备数据和模型 ---
    train_loader, val_loader = setup_dataloaders(
        train_data=train_data_raw,
        val_data=val_data_raw,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        batch_size=BATCH_SIZE,  # (你可以试试用 8 或 16)
        max_allowed_len=MAX_SEQ_LEN
    )

    model, loss_fn, optimizer, start_epoch, best_val_loss, learning_rate = initialize_model(
        DEVICE,
        checkpoint_path=CHECKPOINT_FILE
    )

    # --- (Bug 1 修复: 使用 device.type 并检查) ---
    scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=(DEVICE.type == 'cuda'))

    print(f"\n--- 将从 Epoch {start_epoch} 开始训练... ---")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")

        train_loss, new_best_val_loss = train_one_epoch(
            model, train_loader, val_loader, loss_fn, optimizer,
            DEVICE, scaler, PAD_IDX_SRC, PAD_IDX_TGT,
            epoch=epoch,
            lr_scheduler=learning_rate,
            log_interval=LOG_INTERVAL,
            validate_interval=VALIDATE_INTERVAL,
            best_val_loss=best_val_loss
        )

        best_val_loss = new_best_val_loss

        print(f"--- Epoch {epoch} 平均训练损失: {train_loss:.4e} ---")

        # (我们也保存“最新”的检查点)
        if epoch % 10 == 0:
            print(f"  (正在保存 Epoch {epoch} 的检查点到 {CHECKPOINT_FILE}...)")
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint_data, CHECKPOINT_FILE)

    print("\n--- 训练完成 ---")
    print(f"最好的验证集 Loss: {best_val_loss:.4e}")
    print(f"最好的模型已保存到: transformer_best_model.pth")
