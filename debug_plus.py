import torch
import torch.nn as nn
import torch.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from tokenizers import Tokenizer
from pathlib import Path
import sys
import os
import math
from typing import Callable

# --- 确保能找到我们的模块 ---
sys.path.append(str(Path(__file__).parent))
try:
    from model import Transformer
    from train_utils import create_padding_mask, create_look_ahead_mask
except ImportError as e:
    print(f"错误: 找不到 model.py 或 train_utils.py。 {e}")
    sys.exit(1)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----------------------------------------------------------------------
# 【全局】: 分词器和特殊 Token
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# 【数据管道】: (来自 train.py)
# ----------------------------------------------------------------------

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

        src_ids = self.src_tokenizer.encode(src_text).ids
        tgt_ids = self.tgt_tokenizer.encode(tgt_text).ids

        tgt_input = torch.tensor([SOS_IDX_TGT] + tgt_ids, dtype=torch.long)
        tgt_label = torch.tensor(tgt_ids + [EOS_IDX_TGT], dtype=torch.long)
        src_input = torch.tensor(src_ids, dtype=torch.long)

        return {
            "src_input": src_input,
            "tgt_input": tgt_input,
            "tgt_label": tgt_label
        }


def setup_dataloaders(
        train_data,
        val_data,
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        batch_size: int,
        max_allowed_len: int
):
    print("  正在构建 Torch Dataset...")

    def filter_long(example):
        return len(example['classical']) < max_allowed_len and \
            len(example['modern']) < max_allowed_len

    print(f"  过滤前 (Train): {len(train_data)} / (Val): {len(val_data)}")
    train_data = train_data.filter(filter_long)
    val_data = val_data.filter(filter_long)
    print(f"  过滤后 (Train): {len(train_data)} / (Val): {len(val_data)}")

    train_dataset_torch = TranslationDataset(train_data, src_tokenizer, tgt_tokenizer)
    val_dataset_torch = TranslationDataset(val_data, src_tokenizer, tgt_tokenizer)

    def collate_fn(batch: list):
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
        return src_batch, tgt_inputs, tgt_labels

    train_loader = DataLoader(
        train_dataset_torch,
        batch_size=batch_size,
        shuffle=True,  # (在过拟合测试中, shuffle 意义不大但无害)
        collate_fn=collate_fn,
        num_workers=0,
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
# 【模型与优化器】: (来自 train.py, 保持大样本配置)
# ----------------------------------------------------------------------
def initialize_model(device: torch.device, checkpoint_path: str = None):
    # (!!!) 使用您 train.py 中的大样本配置 (!!!)
    D_MODEL = 512
    NUM_LAYERS = 6
    NUM_HEADS = 8
    D_FF = 2048
    MAX_LEN = 5000
    DROPOUT = 0.0  # (!!!) 保持 Dropout > 0 (!!!)

    model = Transformer(
        SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL,
        NUM_LAYERS, NUM_HEADS, D_FF, MAX_LEN, DROPOUT
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX_TGT, label_smoothing=0.0)

    # (!!!) 保持 train.py 中的 Warmup 配置 (!!!)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = None
    warmup_steps = 2000
    d_model_inv_sqrt = D_MODEL ** (-0.5)

    def lr_scheduler_lambda(step_num):
        step_num += 1
        arg1 = step_num ** (-0.5)
        arg2 = step_num * (warmup_steps ** (-1.5))
        lr_scale_factor = 0.8  # (来自 train.py)
        return d_model_inv_sqrt * min(arg1, arg2) * lr_scale_factor

    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler_lambda)

    start_epoch = 1
    best_val_loss = float('inf')

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"--- 正在从 {checkpoint_path} 加载检查点... ---")
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

        print("--- (正在应用标准 Transformer 初始化: Xavier Uniform) ---")
        for name, p in model.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    print(f"模型已在 {device} 上初始化 (Dropout={DROPOUT})")
    return model, loss_fn, optimizer, start_epoch, best_val_loss, lr_scheduler


# ----------------------------------------------------------------------
# 【诊断函数】: (来自 debug_app.py)
# ----------------------------------------------------------------------

def test_autoregressive_steps(
        model: nn.Module,
        src_tensor: torch.Tensor,
        tgt_tensor: torch.Tensor,  # 完整的真实目标 [SOS, w1, w2, ..., EOS]
        src_mask: torch.Tensor,
        pad_idx_tgt: int,
        create_look_ahead_mask_fn: Callable,
        device: torch.device
):
    """
    (!!!) 关键诊断：逐时间步检查 (来自 debug_app.py) (!!!)
    """
    print("\n" + "=" * 70)
    print("🔬 逐时间步 自回归预测检查 (Autoregressive Step Test)")
    print(f"  (!!!) 当前模型模式: {'TRAIN' if model.training else 'EVAL'} (!!!)")
    print("=" * 70)

    # (此函数已包含 model.eval()，但我们在外部调用时会覆盖它)

    if src_tensor.shape[0] != 1 or tgt_tensor.shape[0] != 1:
        print(f"  [警告] 此测试函数设计为 batch_size=1。检测到 BS={src_tensor.shape[0]}。")
        print("         将强制使用第一个样本 (index 0) 进行测试。")
        src_tensor = src_tensor[:1]
        tgt_tensor = tgt_tensor[:1]
        src_mask = src_mask[:1]

    true_tokens = tgt_tensor[0]
    max_T = true_tokens.shape[0]
    correct_predictions = 0
    total_predictions = max_T - 1

    if total_predictions <= 0:
        print("  [错误] 目标张量太短，无法进行测试。")
        return

    print(f"  将测试 {total_predictions} 个时间步 (T=1 到 T={total_predictions}).")
    print(f"  完整真实序列: {true_tokens.cpu().numpy().tolist()}")
    print("-" * 70)

    try:
        # (!!!) 我们使用 torch.no_grad() 来防止梯度计算,
        # (!!!) 但模型 *本身* 的模式 (train/eval) 是由外部控制的
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            # 1. 编码器只运行一次
            # (我们必须确保 model.encoder 存在)
            enc_output = model.encoder(src_tensor, src_mask)

            # 2. 迭代 T=1 到 T_max-1
            for T in range(1, max_T):
                current_tgt_input = tgt_tensor[:, :T]
                current_tgt_mask = create_look_ahead_mask_fn(current_tgt_input, pad_idx_tgt).to(device)

                # (我们必须确保 model.decoder 存在)
                output_logits_full = model.decoder(current_tgt_input, enc_output, current_tgt_mask, src_mask)

                # (我们必须确保 model.final_proj 存在)
                output_logits = model.final_proj(output_logits_full)

                pred_logits_at_T = output_logits[:, -1, :]
                pred_token_id = pred_logits_at_T.argmax().item()
                true_token_id = true_tokens[T].item()

                is_correct = (pred_token_id == true_token_id)
                if is_correct:
                    correct_predictions += 1

                status_icon = "✅" if is_correct else "❌"
                input_seq_str = current_tgt_input[0].cpu().numpy().tolist()

                print(f"  {status_icon} [T={T:02d}] 输入: {input_seq_str}")
                print(f"         预测: {pred_token_id:5d}  |  真实: {true_token_id:5d}")

                if not is_correct:
                    # (只在第一次失败时打印并跳出)
                    print(f"         (!! 在 T={T} 失败 !!)")
                    break  # (我们只关心它是否完美)

    except Exception as e:
        print(f"\n  [!! 严重错误 !!] 测试在 T={T} 中断: {e}")
        import traceback
        traceback.print_exc()

    print("\n[分析结果]")
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"  总体正确率: {correct_predictions} / {total_predictions} ({accuracy:.2f}%)")

    if correct_predictions == total_predictions:
        print("  🎉 完美! 模型在所有时间步均预测正确 (Teacher Forcing 模式)。")
    else:
        print(f"  🚨 失败! 模型在 T={correct_predictions + 1} 时首次出现预测错误。")

    print("=" * 70 + "\n")
    return accuracy  # (返回准确率)


def diagnose_model_health(model, src_sentence: str, device):
    """(诊断) T=1 数值分析 (来自 debug_app.py)"""
    print("\n" + "=" * 60)
    print("🏥 模型健康诊断 (T=1 数值分析)")
    print("=" * 60)

    global src_tokenizer, tgt_tokenizer
    global PAD_IDX_SRC, PAD_IDX_TGT, SOS_IDX_TGT, EOS_IDX_TGT

    model.eval()  # (此诊断总是在 eval 模式下运行)

    src_ids = src_tokenizer.encode(src_sentence).ids
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    src_mask = create_padding_mask(src_tensor, PAD_IDX_SRC).to(device)
    tgt_input_T1 = torch.tensor([[SOS_IDX_TGT]], dtype=torch.long, device=device)
    tgt_mask_T1 = create_look_ahead_mask(tgt_input_T1, PAD_IDX_TGT).to(device)

    try:
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            enc_output = model.encoder(src_tensor, src_mask)

            dec_output_T1 = model.decoder(tgt_input_T1, enc_output, tgt_mask_T1, src_mask)
            logits_T1 = model.final_proj(dec_output_T1)
            logits_vec = logits_T1[0, 0]

        print("[1] T=1 前向传播模拟 (Top 5):")
        probs = F.softmax(logits_vec, dim=0)
        top5_values, top5_indices = torch.topk(logits_vec, k=5)

        print(f"  Logits统计: mean={logits_vec.mean().item():.4f}, std={logits_vec.std().item():.4f}")
        for rank, (idx, val) in enumerate(zip(top5_indices, top5_values), 1):
            word = tgt_tokenizer.decode([idx.item()])
            prob = probs[idx].item()
            print(f"    {rank}. {word:<10} (id={idx.item():<6}) logit={val.item():7.2f} prob={prob:.4f}")

        predicted_id = top5_indices[0].item()
        if predicted_id == EOS_IDX_TGT:
            print(f"\n  (!!!!!!) 症状确认: 模拟 T=1 预测了 [EOS]! (Logit={top5_values[0].item():.2f})")

        return True

    except Exception as e:
        print(f"  [错误] T=1 前向传播失败: {e}")
        return False


def translate_greedy(
        model,
        src_sentence: str,
        src_tokenizer,
        tgt_tokenizer,
        device,
        pad_idx_src,
        pad_idx_tgt,
        sos_idx_tgt,
        eos_idx_tgt,
        max_len: int = 150
):
    """(诊断) 贪婪解码 (来自 debug_app.py)"""

    print("\n" + "=" * 60)
    print("✍️ 正在运行贪婪翻译 (Greedy Translate)")
    print(f"  (!!!) 当前模型模式: {'TRAIN' if model.training else 'EVAL'} (!!!)")
    print("=" * 60)

    # (此函数强制使用 eval 模式进行解码)
    model.eval()

    src_ids = src_tokenizer.encode(src_sentence).ids
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    src_mask = create_padding_mask(src_tensor, pad_idx_src).to(device)

    try:
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            enc_output = model.encoder(src_tensor, src_mask)
    except Exception as e:
        print(f"  [错误] 编码器失败: {e}")
        return "[翻译失败：编码器错误]"

    output_ids = [sos_idx_tgt]

    for i in range(max_len):
        tgt_tensor = torch.tensor([output_ids], dtype=torch.long, device=device)
        tgt_mask = create_look_ahead_mask(tgt_tensor, pad_idx_tgt).to(device)

        try:
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                dec_output = model.decoder(tgt_tensor, enc_output, tgt_mask, src_mask)
                logits = model.final_proj(dec_output)
        except Exception as e:
            print(f"  [错误] 解码器在步骤 {i + 1} 失败: {e}")
            return f"[翻译失败：解码器在 T={i + 1} 错误]"

        next_token = logits[0, -1, :].argmax().item()

        if i == 0:
            print("  [Infer T=1 Debug] Logits Top 5 (来自贪婪解码):")
            top_k_scores, top_k_indices = torch.topk(logits[0, -1, :], 5)
            for k in range(5):
                score = top_k_scores[k].item()
                idx = top_k_indices[k].item()
                word = tgt_tokenizer.decode([idx])
                print(f"    Rank {k + 1}: {word} (id={idx}) -> {score:.2f}")

        word = tgt_tokenizer.decode([next_token])
        print(f"  [Greedy] 步骤 {i + 1}: {word} (token_id={next_token})")

        if next_token == eos_idx_tgt:
            break
        output_ids.append(next_token)

    print("=" * 60)

    if len(output_ids) > 1:
        return tgt_tokenizer.decode(output_ids[1:])  # (跳过 SOS)
    else:
        return "[翻译为空]"


# ----------------------------------------------------------------------
# 【核心】: 训练循环 (来自 train.py, 已注入诊断)
# ----------------------------------------------------------------------
def evaluate(model, dataloader, loss_fn, device, pad_idx_src, pad_idx_tgt):
    """(来自 train.py)"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for (src_data, tgt_input_data, tgt_label_data) in dataloader:
            src_batch = src_data.to(device)
            tgt_input = tgt_input_data.to(device)
            tgt_label = tgt_label_data.to(device)
            src_mask = create_padding_mask(src_batch, pad_idx_src).to(device)
            tgt_mask = create_look_ahead_mask(tgt_input, pad_idx_tgt).to(device)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(src_batch, tgt_input, src_mask, tgt_mask)
                loss = loss_fn(
                    logits.reshape(-1, logits.shape[-1]),
                    tgt_label.reshape(-1)
                )
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train_one_epoch(model, train_loader, val_loader, loss_fn, optimizer,
                    device, scaler, pad_idx_src, pad_idx_tgt, epoch, lr_scheduler,
                    log_interval, validate_interval, best_val_loss):
    """
    (!!!) 这是 train.py 的 *并行* 训练循环 (!!!)
    (!!!) 我们已注入了 T=1 / T>1 的 Loss 诊断 (!!!)
    """
    model.train()  # (!!!) 关键：保持 .train() 模式 (激活 Dropout)

    total_loss_epoch = 0
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

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):

            # --- [!!!] 这是正确的“并行”训练 (来自 train.py) [!!!] ---
            logits = model(src_batch, tgt_input, src_mask, tgt_mask)

            loss = loss_fn(
                logits.reshape(-1, logits.shape[-1]),
                tgt_label.reshape(-1)
            )
            # --- [!!!] 并行训练结束 [!!!] ---

            # --- [!!!] 新增：T=1 vs T>1 Loss 诊断 (!!!] ---
            with torch.no_grad():
                loss_t1 = loss_fn(
                    logits[:, 0, :],  # (B, V)
                    tgt_label[:, 0]  # (B)
                )
                if logits.size(1) > 1:
                    loss_t_plus = loss_fn(
                        logits[:, 1:, :].reshape(-1, logits.shape[-1]),
                        tgt_label[:, 1:].reshape(-1)
                    )
                else:
                    loss_t_plus = torch.tensor(0.0)
            # --- [!!!] 诊断结束 [!!!] ---

        # (来自 train.py 的标准反向传播)
        optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # scaler.step(optimizer)
        # scaler.update()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        optimizer.zero_grad(set_to_none=True)
        total_loss_epoch += loss.item()

        # (!!!) 注入了 T=1/T>1 和 LR 打印的日志 (!!!)
        if (batch_count + 1) % log_interval == 0:
            # current_lr = lr_scheduler.get_last_lr()[0]
            print(
                f"  [Train] Epoch {epoch}, Batch {batch_count + 1}/{num_batches}, "
                # f"LR: {current_lr:.2e}, "
                f"Total Loss: {loss.item():.4e} "
                f"(T=1 Loss: {loss_t1.item():.2f}, T>1 Loss: {loss_t_plus.item():.4e})"
            )

        if (batch_count + 1) % validate_interval == 0:
            print(f"  (--- 正在运行验证 @ Batch {batch_count + 1} ---)")
            val_loss = evaluate(model, val_loader, loss_fn, device, pad_idx_src, pad_idx_tgt)
            model.train()  # (切回训练模式)
            print(f"  [Validate] Epoch {epoch}, Batch {batch_count + 1}, Val Loss: {val_loss:.4e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"    (新低点! Val Loss 降至 {val_loss:.4e}。正在保存 'transformer_best_model.pth'...)")
                # (我们只保存 state_dict, 保持与 train.py 一致)
                torch.save(model.state_dict(), "transformer_best_model.pth")

    return total_loss_epoch / num_batches, best_val_loss


# ----------------------------------------------------------------------
# 【主程序】: (来自 debug_app.py 的过拟合 + 诊断流程)
# ----------------------------------------------------------------------

if __name__ == '__main__':

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 正在使用设备: {DEVICE} ---")

    # --- 1. 设置超参数 (来自 train.py, 但用于调试) ---
    BATCH_SIZE = 1  # (!!!) 保持 B=1 以便进行清晰的过拟合测试 (!!!)
    NUM_EPOCHS = 200  # (!!!) 运行 200 个 Epoch 来确保能过拟合 (!!!)
    MAX_SEQ_LEN = 100
    LOG_INTERVAL = 5  # (!!!) 增加打印频率 (!!!)
    VALIDATE_INTERVAL = 20  # (!!!) 增加验证频率 (!!!)
    CHECKPOINT_FILE = "transformer_debug.pth"  # (使用新的检查点文件)

    print("\n--- [阶段一] 加载数据集 (过拟合测试模式) ---")

    try:
        dataset_full = load_dataset('xmj2002/Chinese_modern_classical')
    except Exception as e:
        print(f"!! 错误: 无法加载 HuggingFace 数据集: {e}")
        sys.exit(1)

    # --- (!!!) 注入 debug_app.py 的过拟合数据抽取 (!!!) ---
    print("=" * 60)
    print("!!! 警告: 正在执行 [过拟合测试] 模式 !!!")
    try:
        overfit_data = dataset_full['train'].select(range(20))
        print(f"  (已成功抽取 {len(overfit_data)} 条数据用于“背诵”)")

        # (我们用这 20 条数据既做训练又做验证)
        dataset_train_debug = overfit_data
        dataset_val_debug = overfit_data

    except Exception as e:
        print(f"!! 错误: 抽取数据失败: {e}")
        sys.exit(1)

    # --- 2. 准备数据和模型 ---
    train_loader, val_loader = setup_dataloaders(
        train_data=dataset_train_debug,
        val_data=dataset_val_debug,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        batch_size=BATCH_SIZE,
        max_allowed_len=MAX_SEQ_LEN
    )

    model, loss_fn, optimizer, start_epoch, best_val_loss, learning_rate = initialize_model(
        DEVICE,
        checkpoint_path=CHECKPOINT_FILE
    )

    #scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=(DEVICE.type == 'cuda'))
    scaler = None

    print(f"\n--- [阶段二] 开始训练 (使用 train.py 并行循环) ---")
    print(f"--- 将从 Epoch {start_epoch} 开始训练 (共 {NUM_EPOCHS} 个) ---")

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

    print("\n--- [阶段三] 训练完成 ---")
    print(f"最好的验证集 Loss: {best_val_loss:.4e}")

    # ------------------------------------------------------------------
    # (!!!) 注入 debug_app.py 的“终极状态检查” (!!!)
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("--- [阶段四] 开始运行“终极状态检查” ---")
    print("=" * 70)

    try:
        # (1) 准备数据
        print("  (1/4) 准备测试数据 (来自第一个 batch)...")
        src_batch, tgt_input_batch, tgt_label_batch = next(iter(train_loader))

        src_tensor = src_batch[0:1].to(DEVICE)
        tgt_tensor_full = tgt_input_batch[0:1].to(DEVICE)
        src_mask_const = create_padding_mask(src_tensor, PAD_IDX_SRC).to(DEVICE)

        test_sentence_classical = src_tokenizer.decode(src_batch[0].tolist())
        expected_modern = tgt_tokenizer.decode(tgt_label_batch[0].tolist())

        print(f"    测试输入 (文言文): > {test_sentence_classical}")
        print(f"    期望输出 (现代文): > {expected_modern}")

        # (2) 运行 T=1 健康诊断 (总是在 eval 模式)
        print("\n  (2/4) 运行 [diagnose_model_health] (T=1 检查)...")
        diagnose_model_health(model, test_sentence_classical, DEVICE)
        print("\n" + "=" * 40)
        print("🕵️‍♂️ 致命线索检查 (TRAIN_UTILS & DATA)")
        print("=" * 40)

        # 1. 检查 PAD ID 究竟是多少
        real_pad_id = src_tokenizer.token_to_id("<pad>")
        print(f"Tokenier <pad> ID: {real_pad_id}")
        print(f"TrainUtils 默认 PAD_IDX: 1")
        if real_pad_id != 1:
            print("🚨 警告: ID 不匹配！mask = (seq == 1) 可能全错！")

        # 2. 检查 Src Mask 的实际内容 (Cross Attention 的关键)
        # 重新生成一个 mask
        test_src_mask = create_padding_mask(src_tensor, real_pad_id)
        print(f"Src Mask Shape: {test_src_mask.shape}")  # 应该是 (1, 1, 1, S)
        # 打印前 20 个 mask 值 (True=屏蔽, False=保留)
        mask_vals = test_src_mask[0, 0, 0, :20].tolist()
        print(f"Src Mask (前20): {mask_vals}")

        # 3. 逻辑判断
        if all(mask_vals):
            print("❌ 错误: Mask 全为 True！Encoder 被完全屏蔽，模型全盲！")
            print("   原因: 输入数据全是 Pad，或者 pad_idx 搞错了。")
        elif not any(mask_vals) and 1 in src_tensor[0].tolist():
            print("❌ 错误: Mask 全为 False，但输入里明明有 Pad (ID=1)！")
            print("   原因: create_padding_mask 用了错误的 pad_idx。")
        else:
            print("✅ Mask 看起来正常 (混合了 T/F)。")
        print("=" * 40 + "\n")

        # (3) 运行“逐时间步”检查 (在 EVAL 模式)
        print("\n  (3/4) 运行 [test_autoregressive_steps] (在 EVAL 模式)...")
        model.eval()  # (!!!) 强制 EVAL 模式 (!!!)
        acc_eval = test_autoregressive_steps(
            model=model,
            src_tensor=src_tensor,
            tgt_tensor=tgt_tensor_full,
            src_mask=src_mask_const,
            pad_idx_tgt=PAD_IDX_TGT,
            create_look_ahead_mask_fn=create_look_ahead_mask,
            device=DEVICE
        )

        # (4) 运行“逐时间步”检查 (在 TRAIN 模式)
        print("\n  (4/4) 运行 [test_autoregressive_steps] (在 TRAIN 模式)...")
        model.train()  # (!!!) 强制 TRAIN 模式 (!!!)
        acc_train = test_autoregressive_steps(
            model=model,
            src_tensor=src_tensor,
            tgt_tensor=tgt_tensor_full,
            src_mask=src_mask_const,
            pad_idx_tgt=PAD_IDX_TGT,
            create_look_ahead_mask_fn=create_look_ahead_mask,
            device=DEVICE
        )

        # (5) 最终诊断
        print("\n" + "=" * 70)
        print("--- [终极诊断结果] ---")
        print(f"  EVAL 模式 (逐T) 准确率: {acc_eval:.2f}%")
        print(f"  TRAIN 模式 (逐T) 准确率: {acc_train:.2f}%")

        if acc_eval < 100.0 and acc_train > 99.0:
            print("\n  (!!!!!!) 🚨 诊断命中! 🚨 (!!!!!!)")
            print("  模型在 EVAL 模式下失败，但在 TRAIN 模式下成功。")
            print("  这 100% 证明了 Bug 出在某个在 train/eval 模式下")
            print("  行为不一致的模块 (例如 Dropout 或 LayerNorm)。")
        elif acc_eval > 99.0:
            print("\n  (🎉) 恭喜! 模型在 EVAL 模式下通过了测试。")
            print("  “只输出 EOS” 的 Bug 似乎已解决。")
        else:
            print("\n  (???) 诊断未命中。")
            print("  模型在 TRAIN 和 EVAL 模式下均失败。")
            print("  这说明模型根本没有“背诵”成功 (Loss 未降到 0)。")
        print("=" * 70)

        # (6) 运行贪婪翻译 (它会强制 model.eval())
        translation = translate_greedy(
            model=model,
            src_sentence=test_sentence_classical,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            device=DEVICE,
            pad_idx_src=PAD_IDX_SRC,
            pad_idx_tgt=PAD_IDX_TGT,
            sos_idx_tgt=SOS_IDX_TGT,
            eos_idx_tgt=EOS_IDX_TGT,
            max_len=150
        )

        print(f"\n  (!!!) 最终实际翻译结果 (EVAL 模式) (!!!):")
        print(f"  > {translation}")

    except Exception as e:
        print(f"\n  (!!!!!!) 🚨 [终极状态检查] 准备或运行时失败: {e} (!!!!!!)")
        import traceback

        traceback.print_exc()

    print("\n--- 调试脚本运行结束 ---")
