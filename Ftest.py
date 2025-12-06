import torch
import torch.nn.functional as F
import math

# 1. 设置确定性环境，排除随机初始化干扰
torch.manual_seed(42)

# 2. 模拟参数 (Batch=1, Heads=2, SeqLen=4, D_k=8)
B, H, L, D_k = 1, 2, 4, 8
scale = 1.0 / math.sqrt(D_k)

# 3. 构造伪造数据 (模拟 Embedding + Positional Encoding 后的输入)
# 假设 Q, K, V 都是随机生成的
query = torch.randn(B, H, L, D_k, requires_grad=True)
key = torch.randn(B, H, L, D_k, requires_grad=True)
value = torch.randn(B, H, L, D_k, requires_grad=True)

# 4. 构造 Causal Mask (你的实现中使用了 triu)
# [cite: 116] 提到：mask = torch.ones(...).triu(diagonal=1)
# 此时，1 (True) 代表"未来位置"（需要被 Mask 掉），0 (False) 代表"可见位置"
mask_int = torch.ones(L, L).triu(diagonal=1)
mask_bool = mask_int.bool()  # True 表示屏蔽 (Masked)

# 为了广播到 (B, H, L, L)，我们需要 reshape
mask_bool_broadcast = mask_bool.view(1, 1, L, L)

print(f"Mask 形状: {mask_bool_broadcast.shape}")
print("Mask 内容 (True=屏蔽):\n", mask_bool_broadcast[0, 0])

print("-" * 30)


# --- 方案 A: 你的手动实现 (Reference: [cite: 187-208]) ---
def manual_attention(q, k, v, mask):
    # 3.1 计算分数
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # 3.2 应用掩码 (关键点！)
    #  如果是 Bool 掩码(True Masked), 用 -1e9 填充
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask, -1e9)

    # 3.3 Softmax
    attn_probs = F.softmax(attn_scores, dim=-1)

    # 3.4 加权求和
    output = torch.matmul(attn_probs, v)
    return output, attn_probs


out_manual, probs_manual = manual_attention(query, key, value, mask_bool_broadcast)
print("方案 A (手动) 输出范数:", out_manual.norm().item())

print("-" * 30)

# --- 方案 B: F.scaled_dot_product_attention (SDPA) ---
# 嫌疑 1: Mask 逻辑。SDPA 接收 boolean mask 时，True 是代表"保留"还是"屏蔽"？
# 嫌疑 2: is_causal 参数。

print("方案 B (SDPA) 测试:")

try:
    # 尝试 1: 直接传你的 Mask
    out_sdpa_1 = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=~mask_bool_broadcast,
        scale=scale
    )
    print("尝试 1 (传入 Mask) 输出范数:", out_sdpa_1.norm().item())

    # 检查差异
    diff_1 = (out_manual - out_sdpa_1).abs().max()
    print(f"尝试 1 与手写的最大差异: {diff_1.item()}")

except Exception as e:
    print(f"尝试 1 失败: {e}")

print("-" * 20)

try:
    # 尝试 2: 使用 is_causal=True (不传 Mask)
    # SDPA 内部会自动构建 Causal Mask，但这要求你的输入必须是严格的 causal 序列
    out_sdpa_2 = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=None,
        is_causal=True,
        scale=scale
    )
    print("尝试 2 (is_causal=True) 输出范数:", out_sdpa_2.norm().item())

    diff_2 = (out_manual - out_sdpa_2).abs().max()
    print(f"尝试 2 与手写的最大差异: {diff_2.item()}")

except Exception as e:
    print(f"尝试 2 失败: {e}")