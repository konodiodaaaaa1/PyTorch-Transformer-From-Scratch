import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("\n[Sanity Check] 正在加载 model.py [版本: 11_11_CLEAN]\n")
from model_components import (
    PositionalEncoding,
    MultiHeadAttention,
    PositionWiseFeedForward,
    EncoderLayer,
    DecoderLayer
)
from train_utils import create_padding_mask, create_look_ahead_mask


# ----------------------------------------------------------------------
# 模块 1: 编码器 (Encoder)
# ----------------------------------------------------------------------

class Encoder(nn.Module):

    def __init__(self,
                 src_vocab_size: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 max_len: int = 5000,
                 dropout: float = 0.1):
        """
        参数:
            src_vocab_size: 源语言(文言文)的词典大小
            d_model: 模型的总维度 (512)
            num_layers: 编码器层堆叠的数量 (例如 6)
            num_heads: 多头注意力的头数 (8)
            d_ff: 前馈网络的内部维度 (2048)
            max_len: 句子的最大长度 (用于位置编码)
            dropout: Dropout 比率
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # --- 任务 (1): 实例化“零件” ---

        # 1. 词嵌入层
        #    (你的任务)
        #    提示: nn.Embedding(num_embeddings, embedding_dim)
        self.embedding = nn.Embedding(src_vocab_size, self.d_model)

        # 2. 位置编码
        #    (你的任务)
        self.pos_encoding = PositionalEncoding(self.d_model, max_len, dropout)

        # 3. (关键) 堆叠 N 个 EncoderLayer
        #    我们使用 nn.ModuleList 来保存 N 个层
        #    (你的任务)
        #    提示: nn.ModuleList([ ... for _ in range(...) ])
        self.layers = nn.ModuleList(
            [EncoderLayer(self.d_model, num_heads, d_ff, dropout) for _ in range(self.num_layers)])

        # 4. (可选) 编码器最后的归一化
        #    (你的任务)
        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        x 的形状: (batch_size, seq_len)  <-- 注意！输入是 ID
        mask: 填充掩码 (padding mask)
        """

        batch_size = x.size(0)
        seq_len = x.size(1)

        # --- 任务 (2): 实现前向流程 ---

        # 1. (Embedding)
        #    (batch, seq_len) -> (batch, seq_len, d_model)
        #    (你的任务)
        #    提示: 别忘了乘以 sqrt(d_model)，这是论文中的一个小技巧，(使用pre-norm后放弃)
        #    用以缩放嵌入的幅度，使其与位置编码的幅度匹配。
        x = self.embedding(x) * math.sqrt(self.d_model)

        # (!!! DEBUG: 检查 Embedding Norm !!!)
        # if x.size(0) > 0 and x.size(1) > 0:
        #     emb_norm = torch.norm(x[0, 0, :]).item()
        #     print(f"DEBUG: Encoder Emb Norm (Batch 0, T=0): {emb_norm:.4f}")

        # 2. (Positional Encoding)
        #    (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        #    (你的任务)
        x = self.pos_encoding(x)

        # 3. (N x Encoder Layers)
        #    将 x 依次穿过 N 个层
        for layer in self.layers:
            x = layer(x, mask)

        # 4. (Final Norm)
        #    (你的任务)
        x = self.norm(x)

        # (!!! 关键诊断: 检查 Encoder Final Norm !!!)
        # if x.size(0) > 0 and x.size(1) > 0:
        #     final_enc_norm = torch.norm(x[0, 0, :]).item()
        #     print(f"DEBUG: Encoder Final Norm (Batch 0, T=0): {final_enc_norm:.4f}")

        return x


# ----------------------------------------------------------------------
# 模块 2: 解码器 (Decoder)
# ----------------------------------------------------------------------

class Decoder(nn.Module):

    def __init__(self,
                 tgt_vocab_size: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 max_len: int = 5000,
                 dropout: float = 0.1):
        """
        参数:
            tgt_vocab_size: 目标语言(现代文)的词典大小
            (其他参数同 Encoder)
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # --- 任务 (1): 实例化“零件” ---

        # 1. 词嵌入层 (目标语言)
        #    (你的任务)
        self.embedding = nn.Embedding(tgt_vocab_size, self.d_model)

        # 2. 位置编码
        #    (你的任务)
        self.pos_encoding = PositionalEncoding(self.d_model, max_len, dropout)

        # 3. (关键) 堆叠 N 个 DecoderLayer
        #    (你的任务)
        self.layers = nn.ModuleList(
            [DecoderLayer(self.d_model, num_heads, d_ff, dropout) for _ in range(self.num_layers)])

        # 4. 解码器最后的归一化
        #    (你的任务)
        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                x: torch.Tensor,
                enc_output: torch.Tensor,
                look_ahead_mask: torch.Tensor,
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        x (Tensor): 目标(现代文)的 ID, (batch, seq_len_tgt)
        enc_output (Tensor): 编码器的输出, (batch, seq_len_src, d_model)
        look_ahead_mask (Tensor): 前瞻掩码 (用于 MHA 1)
        padding_mask (Tensor): 填充掩码 (用于 MHA 2)
        """

        seq_len = x.size(1)
        # --- 任务 (2): 实现前向流程 ---

        # 1. (Embedding) + 缩放
        #    (batch, seq_len_tgt) -> (batch, seq_len_tgt, d_model)
        #    (你的任务)
        x = self.embedding(x) * math.sqrt(self.d_model)

        # 2. (Positional Encoding) + Dropout
        #    (你的任务)
        x = self.pos_encoding(x)

        # 3. (N x Decoder Layers)
        #    将 x 依次穿过 N 个层
        for idx, layer in enumerate(self.layers):  # <--- 关键修改：添加 idx
            # (注意: DecoderLayer 需要 4 个参数)
            # 我们在这里传入索引作为调试信息
            # if x.size(1) == 1:
            #     print(f"\n[Layer {idx + 1}] --------------------------------")

            x = layer(x, enc_output, look_ahead_mask, padding_mask)

        # 4. (Final Norm) + Dropout
        #    (你的任务)
        x = self.norm(x)

        return x


# ----------------------------------------------------------------------
# 模块 3: 最终的 Transformer
# ----------------------------------------------------------------------

class Transformer(nn.Module):

    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 max_len: int = 5000,
                 dropout: float = 0.1):
        """
        参数:
            src_vocab_size: 源语言(文言文)的词典大小 (来自 src_tokenizer)
            tgt_vocab_size: 目标语言(现代文)的词典大小 (来自 tgt_tokenizer)
            (其他参数是我们熟悉的超参数)
        """
        super().__init__()

        # --- 任务 (1): 实例化两台“整机” ---

        # 1. 实例化“编码器”
        #    (你的任务)
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)

        # 2. 实例化“解码器”
        #    (你的任务)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)

        # 在 __init__ 中添加一个 Norm 层，用于 Encoder Output
        # self.enc_out_norm = nn.LayerNorm(d_model)  # <--- 新增

        # --- 任务 (2): (关键) 实例化“最终输出头” ---
        # 这是一个简单的线性层(Linear Layer)
        # 它的工作是把 Decoder 输出的“思考”向量 (d_model 维)
        # 投影(Project)回 目标词典大小 (tgt_vocab_size 维)
        # 这样我们才能得到每个词的“分数” (Logits)
        # (你的任务)
        self.final_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)

        if self.decoder.embedding.weight.shape == self.final_proj.weight.shape:
            self.final_proj.weight = self.decoder.embedding.weight

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播 (已修复 Bug #22)

        参数:
            src (Tensor): 源(文言文)的 ID, (batch, seq_len_src)
            tgt (Tensor): 目标(现代文)的 ID, (batch, seq_len_tgt)
            src_mask (Tensor): (Encoder) 源填充掩码
            tgt_mask (Tensor): (Decoder) 目标前瞻掩码
        """

        enc_output = self.encoder(src, src_mask)  # <-- Encoder 只需要 src_mask

        # enc_output_normed = self.enc_out_norm(enc_output)

        # (!!! 诊断性强制打印 !!!)
        # if enc_output_normed.size(1) > 0:
        #     print(f"DEBUG: EncOut Norm (Batch 0, T=0): {torch.norm(enc_output_normed[0, 0, :]).item():.4f}")

        dec_output = self.decoder(tgt, enc_output,
                                  look_ahead_mask=tgt_mask,
                                  padding_mask=src_mask)  # <-- (使用传入的 src_mask)

        logits = self.final_proj(dec_output)

        # (!!! 新增修正: Logit 稳定化 - 应对 T=1 爆炸 !!!)
        # if tgt.size(1) == 1:
        #     # T=1 时，强制对 DecOut 进行 Z-score 标准化
        #     dec_output_normed = (dec_output - dec_output.mean(dim=-1, keepdim=True)) / (
        #             dec_output.std(dim=-1, keepdim=True) + 1e-5)
        #
        #     # 使用新的归一化后的 DecOut 重新计算 Logits
        #     # 这一步旨在临时绕过 DecOut 的数值异常，强制 Logit 处于正常范围。
        #     logits = self.final_proj(dec_output_normed)

        return logits

# if __name__ == '__main__':
#
#     print("\n--- 正在测试 Transformer (整机) 模块 ---")
#
#     # --- (关键) 从我们的分词器获取词典大小 ---
#     # (为了运行这个单元测试，你需要导入 Tokenizer 并加载文件)
#     # from tokenizers import Tokenizer
#     # src_tokenizer = Tokenizer.from_file("src_tokenizer.json")
#     # tgt_tokenizer = Tokenizer.from_file("tgt_tokenizer.json")
#     # SRC_VOCAB_SIZE = src_tokenizer.get_vocab_size()
#     # TGT_VOCAB_SIZE = tgt_tokenizer.get_vocab_size()
#
#     # (为了简单，我们先用我们训练时设置的数字)
#     SRC_VOCAB_SIZE = 150000
#     TGT_VOCAB_SIZE = 200000
#
#     # --- 定义模型超参数 ---
#     D_MODEL = 512
#     NUM_LAYERS = 6  # 论文原文是 6
#     NUM_HEADS = 8
#     D_FF = 2048  # 512 * 4
#     MAX_LEN = 5000
#     DROPOUT = 0.1
#
#     # --- 定义输入数据形状 ---
#     BATCH_SIZE = 16
#     SEQ_LEN_SRC = 100  # 文言文 (源) 长度
#     SEQ_LEN_TGT = 90  # 现代文 (目标) 长度
#
#     # 1. 初始化 Transformer (整机)
#     transformer = Transformer(
#         src_vocab_size=SRC_VOCAB_SIZE,
#         tgt_vocab_size=TGT_VOCAB_SIZE,
#         d_model=D_MODEL,
#         num_layers=NUM_LAYERS,
#         num_heads=NUM_HEADS,
#         d_ff=D_FF,
#         max_len=MAX_LEN,
#         dropout=DROPOUT
#     )
#
#     # 2. 创建假的输入 (这一次是 token ID，所以是 torch.randint)
#     fake_src = torch.randint(0, SRC_VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN_SRC))
#     fake_tgt = torch.randint(0, TGT_VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN_TGT))
#
#     # 3. (在真实训练中，掩码是必须的，但形状测试可以传 None)
#
#     # 4. 前向传播
#     output_logits = transformer(fake_src, fake_tgt,
#                                 src_mask=None,
#                                 tgt_mask=None,
#                                 src_padding_mask=None)
#
#     # 5. 检查最终输出的形状
#     expected_shape = (BATCH_SIZE, SEQ_LEN_TGT, TGT_VOCAB_SIZE)
#
#     print(f"输入 (Src) 形状 (ID): {fake_src.shape}")
#     print(f"输入 (Tgt) 形状 (ID): {fake_tgt.shape}")
#     print(f"输出 (Logits) 形状: {output_logits.shape}")
#     print(f"期望的输出形状: {expected_shape}")
#
#     if output_logits.shape == expected_shape:
#         print("\n✅ Transformer (整机) 模块测试成功！")
#     else:
#         print("\n❌ Transformer (整机) 模块测试失败！")
