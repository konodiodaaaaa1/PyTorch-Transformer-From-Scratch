import torch
import torch.nn as nn
import torch.amp as amp
import torch.nn.functional as F
from tokenizers import Tokenizer
from pathlib import Path
import sys
import os
import heapq  # (我们将用“堆”来高效地管理 Beam)

# --- 确保能找到我们的模块 ---
sys.path.append(str(Path(__file__).parent))

try:
    from model import Transformer
    from train_utils import create_padding_mask, create_look_ahead_mask
except ImportError as e:
    print(f"错误: 找不到 model.py 或 train_utils.py。 {e}")
    sys.exit(1)


# ----------------------------------------------------------------------
# 【阶段五：推理 (Inference)】
# 模块 1: Beam Search (集束搜索) 翻译器
# (这 *替换* 了我们之前那个“贪婪”的 translate 函数)
# ----------------------------------------------------------------------

class BeamSearchTranslator:

    def __init__(self, model, src_tokenizer, tgt_tokenizer, device,
                 pad_idx, sos_idx, eos_idx, unk_idx):
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.unk_idx = unk_idx

        self.model.eval()

    def translate(self, src_sentence: str, beam_size: int = 5, max_len: int = 150,
                  length_penalty: float = 0.5) -> str:

        # --- 1. 编码器 (Encoder) ---
        src_ids = self.src_tokenizer.encode(src_sentence).ids
        src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(
            self.device)
        src_mask = create_padding_mask(src_tensor, self.pad_idx).to(self.device)

        with torch.no_grad():
            with amp.autocast(device_type='cuda'):
                enc_output = self.model.encoder(src_tensor, src_mask)
                enc_output = enc_output.expand(beam_size, -1, -1)

        # --- 2. 初始化集束 (Beams) ---
        beams = []

        initial_seq = torch.tensor([[self.sos_idx]], dtype=torch.long, device=self.device)

        # (修正 1: 使用 0 作为初始 counter)
        heapq.heappush(beams, (0.0, 0, initial_seq))

        completed_beams = []

        # --- (关键修正 2: 在循环 *外* 初始化一个 *唯一* 的计数器) ---
        tie_breaker_counter = 0

        # --- 3. 自回归循环 (Decoder) ---
        for i in range(max_len):

            new_beams = []

            current_scores = []
            current_seqs = []
            while beams:
                score, _, seq = heapq.heappop(beams)  # (解包 3 个)
                current_scores.append(score)
                current_seqs.append(seq)

            if not current_seqs:
                break

            tgt_ids_batch = torch.cat(current_seqs, dim=0)

            with torch.no_grad():
                with amp.autocast(device_type='cuda'):
                    tgt_mask = create_look_ahead_mask(tgt_ids_batch, self.pad_idx).to(self.device)
                    dec_output = self.model.decoder(tgt_ids_batch, enc_output[:len(current_seqs)], tgt_mask, src_mask)
                    logits = self.model.final_proj(dec_output)

            last_word_log_probs = F.log_softmax(logits[:, -1, :], dim=1)
            unk_column_index = self.unk_idx
            last_word_log_probs[:, unk_column_index] += -1000.0

            for j in range(len(current_seqs)):
                old_seq = current_seqs[j]
                old_score = current_scores[j]

                top_k_scores, top_k_indices = torch.topk(last_word_log_probs[j], k=beam_size)

                for k in range(beam_size):
                    new_word_id = top_k_indices[k].view(1, 1)
                    new_word_score = top_k_scores[k].item()

                    new_total_score = old_score + new_word_score
                    new_seq = torch.cat([old_seq, new_word_id], dim=1)

                    # --- (关键修正 3: 使用 *唯一* 的计数器) ---
                    heapq.heappush(new_beams, (new_total_score, tie_breaker_counter, new_seq))
                    tie_breaker_counter += 1  # (确保下一次 push 绝对不会打平)



            # 5. "剪枝"
            beams = []
            k_best_items = heapq.nlargest(beam_size, new_beams)

            for item in k_best_items:
                score, counter, seq = item  # 解包 3 个

                if seq[0, -1].item() == self.eos_idx:
                    completed_beams.append((score, seq))
                else:
                    heapq.heappush(beams, item)  # 把 3 个都放回去

        # --- 4. 循环结束 (选择最佳) ---
        for score, _, seq in beams:  # 解包 3 个
            completed_beams.append((score, seq))

        if not completed_beams:
            return "[错误：Beam Search 未能生成任何结果]"

        best_hypothesis = max(
            completed_beams,
            key=lambda h: h[0] / (h[1].size(1) ** length_penalty)
        )

        # --- 5. 转换回文本 ---
        final_ids_list = best_hypothesis[1].cpu().squeeze(0).tolist()

        if len(final_ids_list) > 1:
            translated_text = self.tgt_tokenizer.decode(final_ids_list[1:])
        else:
            translated_text = ""

        return translated_text


# ----------------------------------------------------------------------
# 【主程序：可交互的翻译器】(已更新为使用 BeamSearchTranslator)
# ----------------------------------------------------------------------

if __name__ == '__main__':

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # --- 1. 定义超参数 (使用你“健康”的训练参数) ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SCRIPT_DIR = Path(__file__).parent

    D_MODEL = 512
    NUM_LAYERS = 6
    NUM_HEADS = 8
    D_FF = 2048
    MAX_LEN = 5000
    DROPOUT = 0.1

    # (推理时的新参数)
    BEAM_SIZE = 5  # (K=5 是一个很好的起点)

    # --- 2. 加载分词器和模型权重 ---
    print("正在加载分词器...")
    SRC_TOKENIZER_PATH = SCRIPT_DIR / "src_tokenizer.json"
    TGT_TOKENIZER_PATH = SCRIPT_DIR / "tgt_tokenizer.json"

    try:
        src_tokenizer = Tokenizer.from_file(str(SRC_TOKENIZER_PATH))
        tgt_tokenizer = Tokenizer.from_file(str(TGT_TOKENIZER_PATH))
    except Exception as e:
        print(f"错误: 找不到或无法加载分词器文件。 {e}")
        sys.exit(1)

    SRC_VOCAB_SIZE = src_tokenizer.get_vocab_size()
    TGT_VOCAB_SIZE = tgt_tokenizer.get_vocab_size()

    PAD_IDX = src_tokenizer.token_to_id("<pad>")
    SOS_IDX = tgt_tokenizer.token_to_id("<sos>")
    EOS_IDX = tgt_tokenizer.token_to_id("<eos>")
    UNK_IDX = tgt_tokenizer.token_to_id("<unk>")

    # (替换为你保存的 pth 文件名)
    MODEL_FILE_NAME = "transformer_checkpoint.pth"  # <--- (确保文件名正确)
    MODEL_FILE_PATH = SCRIPT_DIR / MODEL_FILE_NAME

    # 2c. 初始化模型
    print("正在初始化模型架构...")
    model_architecture = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT
    ).to(DEVICE)

    # 2d. 加载训练好的权重
    try:
        print(f"正在从 {MODEL_FILE_PATH} 加载权重...")
        checkpoint = torch.load(MODEL_FILE_PATH, map_location=DEVICE)
        model_architecture.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        print("请检查 MODEL_FILE_NAME 和超参数是否与你训练时完全一致。")
        sys.exit(1)

    # --- (关键) 3. 实例化我们的翻译器 ---
    translator = BeamSearchTranslator(
        model=model_architecture,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        device=DEVICE,
        pad_idx=PAD_IDX,
        sos_idx=SOS_IDX,
        eos_idx=EOS_IDX,
        unk_idx=UNK_IDX
    )

    print("\n--- 翻译器已准备就绪 (Beam Search, K=5) ---")
    print("(输入 'exit' 退出)")

    # --- 4. 可交互的循环 (已修正) ---
    while True:
        try:
            print("\n请输入文言文 (粘贴多行后, 按两次回车 结束输入):")
            lines = []
            while True:
                line = input()

                if line.strip() == "":
                    break
                if line.strip().upper() == "EXIT":
                    print("再见！")
                    sys.exit(0)

                lines.append(line)

            if not lines:
                continue

            src_sentence = " ".join(lines)

            # --- (翻译) ---
            translation = translator.translate(src_sentence, beam_size=BEAM_SIZE)

            # --- 打印结果 ---
            print("\n--- 翻译结果 (Beam Search) ---")
            print(translation)
            print("---------------------------------")

        except KeyboardInterrupt:
            print("\n再见！")
            sys.exit(0)
        except Exception as e:
            print(f"\n[翻译时出错]: {e}")
            import traceback

            traceback.print_exc()