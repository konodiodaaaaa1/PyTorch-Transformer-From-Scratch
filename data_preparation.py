import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import Unigram  # <--- 这是正确的模型
from tokenizers.trainers import UnigramTrainer

# --- 1. 加载数据集 ---
print("正在加载数据集...")
# 第一次运行时，它会自动从 Hugging Face 下载并缓存
dataset = load_dataset('xmj2002/Chinese_modern_classical')

print("数据集加载完毕。结构如下：")
print(dataset)

# --- 2. 检视一条数据 ---
# 让我们看看 'train' 集的第一条数据是什么样的
example = dataset['train'][0]
print("\n数据示例:")
print(example)

print(f"\n文言文: {example['classical']}")
print(f"现代文: {example['modern']}")

# 确定源语言和目标语言的键名
SRC_LANGUAGE_KEY = 'classical'
GT_LANGUAGE_KEY = 'modern'

special_symbols = ["<unk>", "<pad>", "<sos>", "<eos>"]
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


def yield_sentences(data_iter, language_key: str):
    for data_sample in data_iter:
        # 注意：我们这里 yield 整个句子字符串，而不是分词列表
        yield data_sample[language_key]


# --- 创建一个“分词器/词典” ---
def train_tokenizer(data_iterator,  vocab_size=20000):
    # 1. 初始化一个空的 Tokenizer
    tokenizer = Tokenizer(Unigram())

    # 2. 设置训练器
    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=special_symbols,
        unk_token=UNK_TOKEN
    )

    # 3. 训练 (Unigram 可以直接处理原始句子)
    tokenizer.train_from_iterator(
        data_iterator,  # <--- 直接传入原始句子
        trainer=trainer
    )

    # 4. 设置 Padding
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id(PAD_TOKEN),
        pad_token=PAD_TOKEN
    )

    return tokenizer


# --- 6. 分别训练 文言文 和 现代文 的词典 ---
print("\n正在训练源语言 (文言文) 分词器...")
src_iter = yield_sentences(dataset['train'], SRC_LANGUAGE_KEY)
src_tokenizer = train_tokenizer(src_iter, vocab_size=100000)  # 文言文字少
print(f"源语言词典大小: {src_tokenizer.get_vocab_size()}")

print("\n正在训练目标语言 (现代文) 分词器...")
tgt_iter = yield_sentences(dataset['train'], GT_LANGUAGE_KEY)
tgt_tokenizer = train_tokenizer(tgt_iter, vocab_size=200000)  # 现代文字多
print(f"目标语言词典大小: {tgt_tokenizer.get_vocab_size()}")

# --- 7. 测试分词器 ---
print("\n分词器测试 (文言文):")
test_sentence_src = "学而时习之，不亦说乎？"
# 自动完成: 分词 -> 转换为 ID
encoding_src = src_tokenizer.encode(test_sentence_src)

print(f"句子: {test_sentence_src}")
print(f"分词: {encoding_src.tokens}")
print(f"索引 (IDs): {encoding_src.ids}")
# .decode() 自动完成: ID -> 合并为句子
print(f"反转 (句子): {src_tokenizer.decode(encoding_src.ids)}")

print("\n分词器测试 (现代文):")
test_sentence_tgt = "学习并时常温习它。"
encoding_tgt = tgt_tokenizer.encode(test_sentence_tgt)
print(f"句子: {test_sentence_tgt}")
print(f"分词: {encoding_tgt.tokens}")
print(f"索引 (IDs): {encoding_tgt.ids}")
print(f"反转 (句子): {tgt_tokenizer.decode(encoding_tgt.ids)}")

# (可选) 保存我们训练好的分词器，下次就不用再训练了
src_tokenizer.save("src_tokenizer.json")
tgt_tokenizer.save("tgt_tokenizer.json")
print("\n分词器已保存为 .json 文件。")
