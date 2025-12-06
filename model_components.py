import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ----------------------------------------------------------------------
# 组件 2: 位置编码 (Positional Encoding)
# ----------------------------------------------------------------------

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        参数:
            d_model: 词向量的维度 (例如 512)
            max_len: 句子支持的最大长度 (例如 5000)
            dropout: Dropout 的比率
        """
        super().__init__()

        # --- 任务 (1): 计算位置编码矩阵 ---
        # 目标: 创建一个形状为 (max_len, d_model) 的矩阵 `pe`

        # 1. 创建一个全零矩阵用于填充
        pe = torch.zeros(max_len, d_model)

        # 2. 创建一个形状为 (max_len, 1) 的位置张量 `position`
        #    (即 [[0], [1], [2], ..., [max_len-1]])
        position = torch.arange(max_len).unsqueeze(1)  # .unsqueeze(1) 将其变为 (max_len, 1)

        # 3. (关键) 计算公式中的分母部分 `10000^(2i / d_model)`
        #    这可以用 `torch.exp(torch.arange(...) * ...)` 来高效实现
        #    我们只计算偶数维度 (i)，所以步长是 2
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # 4. (关键) 将 sin 和 cos 注入矩阵
        #    偶数索引 (pe[:, 0::2]) 使用 sin
        pe[:, 0::2] = torch.sin(position * div_term)
        #    奇数索引 (pe[:, 1::2]) 使用 cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # 5. (重要) 将 `pe` 增加一个 batch 维度 (1, max_len, d_model)
        #    并注册为 buffer。
        #    我们不希望这个矩阵被视为“模型参数”进行梯度下降（它不是学出来的），
        #    所以我们将其注册为 buffer。
        self.register_buffer('pe', pe.unsqueeze(0))

        # 6. 定义 Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        x 的形状: (batch_size, seq_len, d_model)
        """

        # --- 任务 (2): 将位置编码 "加" 到输入 x ---

        # 1. 从 buffer `pe` 中取出我们需要的前 seq_len 个位置编码
        #    x.size(1) 就是句子的实际长度 (seq_len)
        #    self.pe 的形状是 (1, max_len, d_model)
        #    我们需要的是 (1, seq_len, d_model)
        #    我们设置 .requires_grad_(False) 来明确告诉 PyTorch 不计算它的梯度
        pos_encoding = self.pe[:, :x.size(1), :].requires_grad_(False)

        # 2. 将 x 与位置编码相加
        #    (PyTorch 的广播机制会自动将 (1, seq_len, d_model)
        #     扩展到 (batch_size, seq_len, d_model) 并相加)
        x = x + pos_encoding

        # 3. 应用 dropout
        return self.dropout(x)


# 组件 3: 多头注意力 (Multi-Head Attention)
# ----------------------------------------------------------------------

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        参数:
            d_model: 模型的总维度 (例如 512)
            num_heads: 头的数量 (例如 8)
            dropout: Dropout 比率
        """
        super().__init__()
        # 确保 d_model 可以被 num_heads 整除
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        # --- 任务 (1): 定义所有需要的层 ---

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度 (例如 512 / 8 = 64)

        # 我们需要 4 个全连接层 (Linear)
        # W_q, W_k, W_v 用于将输入 (d_model) 映射到 Q, K, V (d_model)
        # W_o 用于将多头输出 (d_model) 映射回最终输出 (d_model)

        # 提示: nn.Linear(in_features, out_features)
        # W_q, W_k, W_v 都是 d_model -> d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # W_o 也是 d_model -> d_model
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        参数:
            q (Query): 形状 (batch_size, seq_len_q, d_model)
            k (Key):   形状 (batch_size, seq_len_k, d_model)
            v (Value): 形状 (batch_size, seq_len_v, d_model)
                       (通常 seq_len_k 和 seq_len_v 是一样的)
            mask: 掩码 (用于 padding 或 look-ahead)，
                  形状 (batch_size, 1, 1, seq_len_k) 或 (batch_size, 1, seq_len_q, seq_len_k)
        """

        batch_size, seq_len_q, _ = q.shape
        _, seq_len_k, _ = k.shape

        # --- 任务 (2): 拆分 (Split) ---
        # 1. 将 (batch, seq_len, d_model) 投影到 (batch, seq_len, d_model)
        #    (你的任务)
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        # 2. (关键) 拆分成多头
        #    将 (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        #    这需要 .view() 和 .transpose()
        #    (你的任务)
        #    提示: Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)

        # --- 任务 (3): 手动实现缩放点积注意力 (白盒模式) ---
        # 替换原有的 F.scaled_dot_product_attention，显式处理矩阵运算和掩码

        # 3.1 计算注意力分数 (Scaled Dot-Product)
        # Q: (B, H, L, D_k), K^T: (B, H, D_k, S) -> Scores: (B, H, L, S)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3.2 应用掩码 (手动处理，绝对透明)
        if mask is not None:
            # 你的 mask 可能是 (B, 1, 1, S) 或 (B, 1, L, S)
            # PyTorch 的广播机制会自动将 mask 广播到 (B, H, L, S)

            # 鲁棒性检查：处理 Bool 类型 (True=屏蔽) 和 Float 类型 (负无穷=屏蔽)
            if mask.dtype == torch.bool:
                # 如果是 Bool 掩码 (True=Masked)，用 -1e9 填充 (模拟 -inf)
                attn_scores = attn_scores.masked_fill(mask, -1e4)
            else:
                # 如果是 0/-inf 掩码，直接相加
                attn_scores = attn_scores + mask

        # 3.3 Softmax 和 Dropout
        # 在最后一个维度 (Seq_len_k) 上进行归一化
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 应用 Dropout (仅在训练时)
        attn_probs = self.dropout(attn_probs)

        # 3.4 加权求和
        # Probs: (B, H, L, S), V: (B, H, S, D_k) -> Output: (B, H, L, D_k)
        x = torch.matmul(attn_probs, V)

        # x 的形状是 (batch, num_heads, seq_len_q, d_k)

        # --- 任务 (4): 合并 (Concatenate) ---
        #    将多头拼接回来
        #    (batch, num_heads, seq_len_q, d_k) -> (batch, seq_len_q, num_heads, d_k)
        #    这需要 .transpose(1, 2)
        x = x.transpose(1, 2).contiguous()

        # 2. 合并: (batch, seq_len_q, num_heads, d_k) -> (batch, seq_len_q, d_model)
        #    这需要 .contiguous().view()
        x = x.contiguous().view(batch_size, -1, self.d_model)

        # 3. 通过最后的输出层 W_o
        output = self.W_o(x)

        return output


# 组件 4: 位置前馈网络 (Position-wise Feed-Forward Network)
class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        参数:
            d_model: 模型的总维度 (例如 512)
            d_ff: 内部隐藏层的维度 (例如 2048)
            dropout: Dropout 比率
        """
        super().__init__()

        # --- 任务 (1): 定义两个 Linear 层和一个 Dropout ---
        # self.linear_1 = nn.Linear(..., ...)
        # self.linear_2 = nn.Linear(..., ...)
        # self.dropout = nn.Dropout(...)

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # 定义激活函数
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        x 的形状: (batch_size, seq_len, d_model)
        """

        # --- 任务 (2): 实现 x -> linear_1 -> relu -> dropout -> linear_2 ---
        # (你的任务)
        # 提示:
        # x = self.linear_1(x)
        # x = self.relu(x)
        # ...

        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)  # 论文中 dropout 在 relu 之后
        x = self.linear_2(x)

        return x


# 组件 5: 编码器层 (Encoder Layer)
class EncoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        参数:
            d_model: 模型的总维度 (例如 512)
            num_heads: 多头注意力的头数 (例如 8)
            d_ff: 前馈网络的内部维度 (例如 2048)
            dropout: Dropout 比率
        """
        super().__init__()

        # --- 任务 (1): 实例化我们所有的“零件” ---

        # 1. 实例化“多头注意力引擎”
        #    (你的任务)
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        # 2. 实例化“前馈加工单元”
        #    (你的任务)
        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # 3. 实例化两个“层归一化” (LayerNorm) 模块
        #    LayerNorm 是 Transformer 成功的关键
        #    (你的任务)
        #    提示: nn.LayerNorm(d_model)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        # 4. 实例化两个 Dropout (用于残差连接之后)
        #    (你的任务)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        x 的形状: (batch_size, seq_len, d_model)
        mask: 编码器掩码 (用于忽略 <pad> 标记)
        """

        # 1. 第一个子层: MHA (Multi-Head Attention)
        # 流程: Norm -> MHA -> Add (残差)

        # Pre-Norm: 先归一化输入
        x_norm = self.norm_1(x)

        # MHA: 将归一化后的输入送入 MHA
        # 注意: Q, K, V 都来自 x_norm
        attn_output = self.mha(x_norm, x_norm, x_norm, mask=mask)

        # Add (残差连接): 将 MHA 输出加到原始输入 x 上
        x = x + self.dropout_1(attn_output)

        # 2. 第二个子层: FFN (Position-wise Feed-Forward)
        # 流程: Norm -> FFN -> Add (残差)

        # Pre-Norm: 对 MHA 之后的 x 进行归一化
        x_norm = self.norm_2(x)

        # FFN: 将归一化后的输入送入 FFN
        ffn_output = self.ffn(x_norm)

        # Add (残差连接): 将 FFN 输出加到中间结果 x 上
        x = x + self.dropout_2(ffn_output)

        return x


# 组件 6: 解码器层 (Decoder Layer)
class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        参数: (同 EncoderLayer)
        """
        super().__init__()

        # --- 任务 (1): 实例化所有“零件” ---

        # 1. 实例化“带掩码的自注意力” (MHA 1)
        #    (你的任务)
        self.mha_self = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        # 2. 实例化“交叉注意力” (MHA 2)
        #    (你的任务)
        self.mha_cross = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        # 3. 实例化“前馈加工单元”
        #    (你的任务)
        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # 4. 实例化 三个“层归一化” (LayerNorm) 模块
        #    (你的任务)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        # (!!! 关键修复: 添加 K, V 的归一化层 !!!)
        self.norm_cross_kv = nn.LayerNorm(d_model)  # <--- 新增

        self.norm_3 = nn.LayerNorm(d_model)

        # 5. 实例化 三个 Dropout
        #    (你的任务)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        # === 新增: T=1时的可学习偏置 ===
        # self.t1_bias = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self,
                x: torch.Tensor,
                enc_output: torch.Tensor,
                look_ahead_mask: torch.Tensor,
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        参数:
            x (Tensor): 解码器的输入, (batch, seq_len_tgt, d_model)
            enc_output (Tensor): 编码器的输出 (K, V), (batch, seq_len_src, d_model)
            look_ahead_mask (Tensor): 前瞻掩码 (用于 MHA 1)
            padding_mask (Tensor): 填充掩码 (用于 MHA 2)
        """

        # if x.size(1) == 1:
        #     x_norm_input = torch.norm(x[0, 0, :]).item()
        #     print(f"[DEBUG-Layer] Input Norm: {x_norm_input:.4f}")  # 始终打印输入范数

        # 1. 步骤 1: 带掩码的自注意力 (MHA 1)
        x_old_self = x  # 保存原始输入 x
        # 1. 步骤 1: 带掩码的自注意力 (MHA 1)
        # 流程: Norm -> MHA -> Add (残差)

        # Pre-Norm: 先归一化输入
        x_norm_1 = self.norm_1(x)

        # MHA Self-Attention: Q, K, V 都来自 x_norm_1
        attn_self_output = self.mha_self(x_norm_1, x_norm_1, x_norm_1, mask=look_ahead_mask)

        # Add (残差): 将 MHA 输出加到原始输入 x 上
        x = x + self.dropout_1(attn_self_output)

        # 2. 步骤 2: 交叉注意力 (MHA 2)
        # 流程: Norm -> MHA -> Add (残差)

        # Pre-Norm: 归一化 Self-Attention 之后的 x
        x_old_cross = x  # 保存原始输入 x
        x_norm_2 = self.norm_2(x)

        # (!!! 关键修正: K 和 V 必须来自归一化的 Encoder Output !!!)
        # 我们使用新的 norm_cross_kv 层来归一化 enc_output
        enc_output_normed_kv = self.norm_cross_kv(enc_output)  # <--- 新增

        # MHA Cross-Attention: Q=x_norm_2, K=enc_output, V=enc_output
        attn_cross_output = self.mha_cross(x_norm_2, enc_output_normed_kv, enc_output_normed_kv, mask=padding_mask)

        # Add (残差): 将 MHA 输出加到中间结果 x 上
        x = x + self.dropout_2(attn_cross_output)

        # 3. 步骤 3: 前馈网络 (FFN)
        # 流程: Norm -> FFN -> Add (残差)
        x_old_ffn = x

        # Pre-Norm: 归一化 Cross-Attention 之后的 x
        x_old_ffn = x  # 保存原始输入 x
        x_norm_3 = self.norm_3(x)

        # FFN: 将归一化后的输入送入 FFN
        ffn_output = self.ffn(x_norm_3)

        # Add (残差): 将 FFN 输出加到中间结果 x 上
        x = x + self.dropout_3(ffn_output)

        # if x.size(0) > 0 and x.size(1) > 0:
        #     current_norm = torch.norm(x[0, x.size(1) - 1, :]).item()
        #
        #     # 注意: 我们无法确定这是 T=1 还是 T=N，但我们打印出来看累积模式
        #     print(f"[UNCONDITIONAL-NORM] SeqLen={x.size(1):<2}, Norm={current_norm:.4f}")

        return x

# --- 单元测试位置编码 (Positional Encoding)  ---
# 我们可以直接在这个文件里运行它，来测试这个模块是否工作正常
# if __name__ == '__main__':
#
#     # 定义超参数
#     d_model = 512  # 词向量维度
#     seq_len = 100  # 假设句子长度为 100
#     batch_size = 16  # 假设批量大小为 16
#     max_len_test = 5000  # 模块支持的最大长度
#
#     print("--- 正在测试 PositionalEncoding 模块 ---")
#
#     # 1. 创建一个假的词向量输入
#     #    (这模拟了【组件1：Embedding】层的输出)
#     #    形状: (batch_size, seq_len, d_model)
#     fake_input = torch.randn(batch_size, seq_len, d_model)
#
#     # 2. 初始化位置编码模块
#     pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len_test)
#
#     # 3. 前向传播
#     output = pos_encoder(fake_input)
#
#     print(f"输入形状: {fake_input.shape}")
#     print(f"输出形状: {output.shape}")
#     print(f"输入和输出的形状是否一致: {fake_input.shape == output.shape}")
#
#     if fake_input.shape == output.shape:
#         print("\n✅ 位置编码模块测试成功！")
#     else:
#         print("\n❌ 位置编码模块测试失败！")

# --- 单元测试多头注意力 (Multi-Head Attention)---
# if __name__ == '__main__':
#
#
#     print("\n--- 正在测试 MultiHeadAttention 模块 ---")
#
#     d_model = 512
#     num_heads = 8
#     seq_len = 100
#     batch_size = 16
#
#     # 1. 初始化 MHA 模块
#     mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
#
#     # 2. 创建假的 Q, K, V 输入 (在 Encoder 中, Q, K, V 来自同一个源)
#     fake_input = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)
#
#     # (在 Decoder 中, K 和 V 可能来自 Encoder, Q 来自 Decoder)
#
#     # 3. 前向传播 (self-attention)
#     output = mha(fake_input, fake_input, fake_input, mask=None)
#
#     print(f"输入形状: {fake_input.shape}")
#     print(f"输出形状: {output.shape}")
#     print(f"输入和输出的形状是否一致: {fake_input.shape == output.shape}")
#
#     if fake_input.shape == output.shape:
#         print("\n✅ 多头注意力模块测试成功！")
#     else:
#         print("\n❌ 多头注意力模块测试失败！")

# --- 单元测试编码器层 (Encoder Layer)---
# if __name__ == '__main__':
#     # ... (之前的 PE, MHA, FFN 测试代码) ...
#
#     print("\n--- 正在测试 EncoderLayer 模块 ---")
#
#     d_model = 512
#     num_heads = 8
#     d_ff = 2048
#     seq_len = 100
#     batch_size = 16
#
#     # 1. 初始化 EncoderLayer 模块
#     encoder_layer = EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
#
#     # 2. 创建一个假的输入
#     fake_input = torch.randn(batch_size, seq_len, d_model)
#
#     # 3. 前向传播
#     output = encoder_layer(fake_input, mask=None)
#
#     print(f"输入形状: {fake_input.shape}")
#     print(f"输出形状: {output.shape}")
#     print(f"输入和输出的形状是否一致: {fake_input.shape == output.shape}")
#
#     if fake_input.shape == output.shape:
#         print("\n✅ 编码器层模块测试成功！")
#     else:
#         print("\n❌ 编码器层模块测试失败！")

# --- 单元测试解码器层 (Decoder Layer)---
# if __name__ == '__main__':
#     # ... (之前的 PE, MHA, FFN, EncoderLayer 测试代码) ...
#
#     print("\n--- 正在测试 DecoderLayer 模块 ---")
#
#     d_model = 512
#     num_heads = 8
#     d_ff = 2048
#     seq_len_tgt = 90  # 假设目标(现代文)句子长度为 90
#     seq_len_src = 100  # 假设源(文言文)句子长度为 100
#     batch_size = 16
#
#     # 1. 初始化 DecoderLayer 模块
#     decoder_layer = DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
#
#     # 2. 创建假的输入
#     fake_tgt_input = torch.randn(batch_size, seq_len_tgt, d_model)
#     fake_src_output = torch.randn(batch_size, seq_len_src, d_model)
#
#     # (在真实训练中, 掩码是必须的, 但在形状测试中我们可以传入 None)
#
#     # 3. 前向传播
#     output = decoder_layer(fake_tgt_input, fake_src_output,
#                            look_ahead_mask=None, padding_mask=None)
#
#     print(f"输入 (Tgt) 形状: {fake_tgt_input.shape}")
#     print(f"输入 (Src) 形状: {fake_src_output.shape}")
#     print(f"输出形状: {output.shape}")
#     print(f"输入(Tgt)和输出的形状是否一致: {fake_tgt_input.shape == output.shape}")
#
#     if fake_tgt_input.shape == output.shape:
#         print("\n✅ 解码器层模块测试成功！")
#     else:
#         print("\n❌ 解码器层模块测试失败！")
