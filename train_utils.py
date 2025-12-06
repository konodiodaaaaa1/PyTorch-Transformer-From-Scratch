import torch
import torch.nn as nn

PAD_IDX = 1  # 假设 <pad> 的 ID 是 1


# ----------------------------------------------------------------------
# 【阶段五：系统运行】
# 任务 5.1: 掩码 (Masking)
# ----------------------------------------------------------------------

def create_padding_mask(seq: torch.Tensor, pad_idx: int = PAD_IDX) -> torch.Tensor:
    """
    创建填充掩码 (Padding Mask)

    参数:
        seq (Tensor): 输入序列 (token ID), 形状 (batch_size, seq_len)
        pad_idx (int): <pad> 标记的 ID

    返回:
        mask (Tensor): 形状 (batch_size, 1, 1, seq_len)
                      或 (batch_size, seq_len)，具体取决于MHA实现。
                      F.scaled_dot_product_attention 接受 (batch, num_heads, seq_len, seq_len)
                      或可以被“广播”(broadcast) 到那个形状的形状。

                      (batch_size, 1, 1, seq_len) 是最安全、最通用的广播形状。
    """

    # 1. 检查 seq 中哪些位置等于 pad_idx
    #    (你的任务)
    #    提示: seq == pad_idx
    #    这将返回一个布尔张量, 形状 (batch_size, seq_len)
    #    (False, False, True, True)
    seq_mask = (seq == pad_idx)
    # 2. (关键) 调整形状以适应 MHA 的广播机制
    #    MHA 期望的掩码形状是 (batch, num_heads, seq_len_q, seq_len_k)
    #    我们通过在 dim 1 和 dim 2 插入维度，将其变为 (batch, 1, 1, seq_len)
    #    这样它就可以自动广播到 (batch, num_heads, seq_len_q, seq_len)
    #    (你的任务)
    #    提示: .unsqueeze(1).unsqueeze(2)
    mask = seq_mask.unsqueeze(1).unsqueeze(2)

    return mask


def create_look_ahead_mask(seq: torch.Tensor, pad_idx: int = PAD_IDX) -> torch.Tensor:
    """
    创建前瞻掩码 (Look-ahead Mask)，它是一个“复合”掩码，
    同时包含了“填充掩码”和“未来词掩码”。

    参数:
        seq (Tensor): 输入序列 (token ID), 形状 (batch_size, seq_len)
        pad_idx (int): <pad> 标记的 ID

    返回:
        mask (Tensor): 最终的复合掩码, 形状 (batch_size, 1, seq_len, seq_len)
                      (F.scaled_dot_product_attention 接受 (N, H, L, S)
                       或可广播的形状，我们用 (N, 1, L, L) 来广播到 H)
    """

    batch_size, seq_len = seq.shape

    # 1. Padding 部分
    padding_mask = create_padding_mask(seq, pad_idx)

    # 2. Causal 部分 (上三角)
    look_ahead_mask = torch.ones((seq_len, seq_len), device=seq.device).triu(diagonal=1).bool()

    # 3. 显式合并 (使用 max 替代 | 以防万一)
    # (Batch, 1, 1, Seq) vs (1, 1, Seq, Seq) -> (Batch, 1, Seq, Seq)
    return torch.max(padding_mask, look_ahead_mask.unsqueeze(0).unsqueeze(0))


# ---  单元测试pad ---
# if __name__ == '__main__':
#     print("--- 正在测试 掩码函数 ---")
#
#     # 1. 创建假的 token ID (batch_size=2, seq_len=5)
#     fake_seq = torch.tensor([
#         [10, 20, 30, PAD_IDX, PAD_IDX],  # 句子 1 (有 2 个 padding)
#         [12, 22, 32, 42, PAD_IDX]  # 句子 2 (有 1 个 padding)
#     ])
#
#     # 2. 调用函数
#     padding_mask = create_padding_mask(fake_seq, PAD_IDX)
#
#     # 3. 检查结果
#     print(f"输入序列 (IDs):\n{fake_seq}")
#     print(f"输入形状: {fake_seq.shape}")
#     print(f"输出掩码 (Mask):\n{padding_mask}")
#     print(f"输出形状: {padding_mask.shape}")
#
#     # 期望的输出形状 (batch, 1, 1, seq_len) -> (2, 1, 1, 5)
#     expected_shape = (fake_seq.size(0), 1, 1, fake_seq.size(1))
#
#     # 期望的掩码内容 (False=保留, True=屏蔽)
#     # tensor([[[[False, False, False,  True,  True]]],
#     #         [[[False, False, False, False,  True]]]])
#     expected_mask_content = torch.tensor([
#         [[[False, False, False, True, True]]],
#         [[[False, False, False, False, True]]]
#     ])
#
#     if padding_mask.shape == expected_shape and torch.equal(padding_mask, expected_mask_content):
#         print("\n✅ 填充掩码 (Padding Mask) 函数测试成功！")
#     else:
#         print("\n❌ 填充掩码 (Padding Mask) 函数测试失败！")

# ---  单元测试与上三角掩码合并 ---
# if __name__ == '__main__':
#     # ... (你之前的 padding_mask 测试) ...
#
#     print("\n--- 正在测试 Look-ahead 掩码函数 ---")
#
#     # 1. 创建假的 token ID (batch_size=2, seq_len=5)
#     fake_seq_tgt = torch.tensor([
#         [10, 20, 30, PAD_IDX, PAD_IDX],  # 句子 1 (有 2 个 padding)
#         [12, 22, 32, 42, PAD_IDX]  # 句子 2 (有 1 个 padding)
#     ])
#
#     # 2. 调用函数
#     look_ahead_mask = create_look_ahead_mask(fake_seq_tgt, PAD_IDX)
#
#     # 3. 检查结果
#     print(f"输入序列 (IDs):\n{fake_seq_tgt}")
#     print(f"输出掩码 (Mask):\n{look_ahead_mask}")
#     print(f"输出形状: {look_ahead_mask.shape}")
#
#     # 期望的形状 (B, 1, S, S) -> (2, 1, 5, 5)
#     expected_shape = (fake_seq_tgt.size(0), 1,
#                       fake_seq_tgt.size(1), fake_seq_tgt.size(1))
#
#     # 期望的掩码内容 (T = 屏蔽)
#     # 1. 未来词掩码 (上三角)
#     # [[F, T, T, T, T],
#     #  [F, F, T, T, T],
#     #  [F, F, F, T, T],
#     #  [F, F, F, F, T],
#     #  [F, F, F, F, F]]
#     # 2. 填充掩码 (最后两列 / 最后一列)
#     # (B=0): [[F, F, F, T, T], ... ] (广播)
#     # (B=1): [[F, F, F, F, T], ... ] (广播)
#     # 3. 两者 "或" (|) 之后的结果:
#
#     expected_mask_b0 = torch.tensor([
#         [False, True, True, True, True],  # Q 0 (10):    Mask future
#         [False, False, True, True, True],  # Q 1 (20):    Mask future
#         [False, False, False, True, True],  # Q 2 (30):    Mask future
#         [False, False, False, True, True],  # Q 3 (PAD):   Mask future + K pads
#         [False, False, False, True, True]  # Q 4 (PAD):   Mask future + K pads
#     ])
#
#     # --- (这是修正后的 b1) ---
#     expected_mask_b1 = torch.tensor([
#         [False, True, True, True, True],  # Q 0 (12):    Mask future
#         [False, False, True, True, True],  # Q 1 (22):    Mask future
#         [False, False, False, True, True],  # Q 2 (32):    Mask future
#         [False, False, False, False, True],  # Q 3 (42):    Mask future
#         [False, False, False, False, True]  # Q 4 (PAD):   Mask future + K pad
#     ])
#
#     # 组合成 (B, 1, S, S)
#     expected_mask_content = torch.stack([expected_mask_b0, expected_mask_b1]).unsqueeze(1)
#
#     if look_ahead_mask.shape == expected_shape and torch.equal(look_ahead_mask, expected_mask_content):
#         print("\n✅ 前瞻掩码 (Look-ahead Mask) 函数测试成功！")
#     else:
#         print("\n❌ 前瞻掩码 (Look-ahead Mask) 函数测试失败！")
#         print("--- 期望的掩码 ---")
#         print(expected_mask_content)

# if __name__ == '__main__':
#     SOS_IDX = 2
#     single_token = torch.tensor([[SOS_IDX]])  # 假设SOS_IDX不是PAD_IDX
#     mask_single = create_look_ahead_mask(single_token)
#     print(f"Single token mask shape: {mask_single.shape}")
#     print(f"Single token mask: {mask_single}")
#
#     # 测试多token序列
#     multi_token = torch.tensor([[SOS_IDX, 100, 200, PAD_IDX, PAD_IDX]])
#     mask_multi = create_look_ahead_mask(multi_token)
#     print(f"Multi token mask shape: {mask_multi.shape}")
#     print(f"Multi token mask[0,0]:\n{mask_multi[0, 0]}")
