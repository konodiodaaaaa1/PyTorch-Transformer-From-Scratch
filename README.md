# TransformerFromScratch：从零实现Transformer的完整学习项目

## 🎯 项目简介

这是一个**从零开始完全手动实现**的Transformer模型项目，专为**文言文到现代文翻译**任务设计。项目不仅仅是简单调用PyTorch内置的`nn.Transformer`模块，而是深入实现了Transformer的每个核心组件，并通过大量调试解决了数值稳定性等深层次问题。

**项目特点**：
- ✅ **完全手写实现**：不使用`nn.Transformer`等高级封装
- ✅ **详细注释**：每行代码都有详细解释，适合学习
- ✅ **深度调试**：包含完整的调试工具和问题解决方案
- ✅ **文言文翻译**：针对中文古典文献翻译任务
- ✅ **工程完整**：包含数据准备、训练、推理全流程

## 🚀 快速开始

### 环境要求

- **Python**: 3.8+
- **PyTorch**: 1.12+（建议2.0+）
- **CUDA**: 可选，建议11.8+（如有NVIDIA GPU）

### 安装指南

#### 方式一：使用Conda（推荐）

```bash
# 1. 创建并激活Conda环境
conda create -n transformer-scratch python=3.11
conda activate transformer-scratch

# 2. 安装PyTorch（根据你的CUDA版本选择）
# CUDA 12.1（RTX 40系列推荐）
conda install pytorch==2.5.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# CUDA 11.8
# conda install pytorch==2.5.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# CPU版本（无GPU）
# conda install pytorch==2.5.1 torchvision torchaudio cpuonly -c pytorch

# 3. 安装项目依赖
pip install transformers==4.31.0 datasets==2.13.1 tokenizers==0.13.3
pip install sacrebleu==2.3.1 nltk==3.8.1 tensorboard==2.13.0
pip install numpy scipy tqdm matplotlib
```

#### 方式二：使用requirements.txt

```bash
# 安装PyTorch（先安装PyTorch，因为需要指定CUDA版本）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

#### 方式三：Docker（高级用户）

```bash
# 构建Docker镜像
docker build -t transformer-scratch .

# 运行容器
docker run -it --gpus all transformer-scratch
```

### 验证安装

```bash
# 验证PyTorch安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import transformers; import datasets; import tokenizers; print('核心库安装成功！')"
```

## 📁 项目结构

```
TransformerFromScratch/
├── model_components.py    # 核心组件：位置编码、多头注意力、前馈网络等
├── model.py              # 完整Transformer架构：编码器、解码器
├── train.py              # 主训练脚本（支持大规模训练）
├── data_preparation.py   # 数据预处理与分词器训练
├── train_utils.py        # 工具函数：掩码生成
├── debug_plus.py         # 高级诊断工具（过拟合测试、数值分析）
├── translate.py          # 推理脚本（支持集束搜索）
├── Ftest.py              # 注意力机制对比测试
├── requirements.txt      # Python依赖包列表
├── environment.yml       # Conda环境配置文件
├── README.md            # 项目说明文档
├── src_tokenizer.json   # 源语言分词器（运行后生成）
├── tgt_tokenizer.json   # 目标语言分词器（运行后生成）
└── Transformer项目经验总结与笔记.pdf  # 完整技术文档
```

## 📊 数据集

本项目使用文言文-现代文平行语料库：
- **数据集**: `xmj2002/Chinese_modern_classical`（自动从HuggingFace下载）
- **规模**: 约50万对句子
- **语言**: 文言文 ↔ 现代中文
- **示例**:
  - 文言文: "学而时习之，不亦说乎？"
  - 现代文: "学习并时常温习它，不也很愉快吗？"

## 🛠️ 使用流程

### 1. 数据准备

```bash
# 首次运行需要下载数据集并训练分词器
python data_preparation.py
```

这将：
- 从HuggingFace下载数据集
- 训练文言文和现代文的分词器
- 保存分词器为JSON文件

### 2. 调试与验证

```bash
# 在20条数据上运行过拟合测试
python debug_plus.py
```

这个脚本会：
- 测试模型能否在小数据集上过拟合
- 验证所有组件正常工作
- 检查数值稳定性
- 提供详细的诊断信息

### 3. 完整训练

```bash
# 使用完整数据集训练模型
python train.py
```

**关键训练参数**：
- **Batch Size**: 8-32（根据GPU内存调整）
- **学习率**: 使用Warmup策略，最大1e-4
- **Epochs**: 100-1000（取决于数据集大小）
- **梯度裁剪**: 1.0
- **Dropout**: 0.1

### 4. 推理测试

```bash
# 启动交互式翻译器
python translate.py
```

支持两种推理模式：
- **贪婪解码**: 简单快速
- **集束搜索**（Beam Search）: 质量更好，支持beam_size参数

## 🔬 技术细节

### 核心实现特点

#### 1. **完全手写的注意力机制**
```python
# 手动实现缩放点积注意力
attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
if mask is not None:
    attn_scores = attn_scores.masked_fill(mask, -1e4)  # 手动处理掩码
attn_probs = F.softmax(attn_scores, dim=-1)
output = torch.matmul(attn_probs, V)
```

#### 2. **Pre-Norm架构**
采用更稳定的Pre-Norm而非原始Post-Norm：
```python
# Pre-Norm: 先归一化再进入子层
x_norm = self.norm_1(x)
attn_output = self.mha(x_norm, x_norm, x_norm, mask=mask)
x = x + self.dropout_1(attn_output)  # 残差连接
```

#### 3. **数值稳定性优化**
- Embedding层乘以√d_model平衡位置编码
- LayerNorm的特殊初始化策略
- 梯度裁剪防止梯度爆炸
- T=1时刻的单独监控

### 关键配置参数

```python
# 模型超参数
D_MODEL = 512          # 模型维度
NUM_LAYERS = 6         # 编码器/解码器层数
NUM_HEADS = 8          # 注意力头数
D_FF = 2048            # 前馈网络隐藏维度
DROPOUT = 0.1          # Dropout率
MAX_LEN = 5000         # 最大序列长度

# 训练参数
BATCH_SIZE = 32        # 批次大小
LEARNING_RATE = 1e-4   # 学习率
WARMUP_STEPS = 4000    # 学习率warmup步数
LABEL_SMOOTHING = 0.1  # 标签平滑
```

## 🧪 调试与诊断

项目包含完整的调试工具链：

### 1. **数值稳定性监控**
- 各层激活值范数检查
- 梯度流向可视化
- T=1特殊时刻的单独监控

### 2. **过拟合测试**
在20条数据上测试模型能否完美记忆：
```bash
python debug_plus.py
```

### 3. **训练诊断**
- 分离T=1和T>1的Loss
- 学习率调度可视化
- 权重分布统计

## 📈 实验结果

### 训练指标
- **训练Loss**: 可收敛到0.1以下
- **验证Loss**: 稳定在1.5-2.0之间
- **推理准确率**: 在小数据集上可达100%过拟合

### 翻译质量
- **BLEU Score**: 待计算（需要参考翻译）
- **人工评估**: 文言文翻译基本准确，能处理常见句式

## 💡 学习收获

通过本项目，你可以深入掌握：

### 1. **Transformer核心机制**
- 注意力机制（QKV范式）的数学原理
- 位置编码的几何意义
- 多头注意力的并行计算

### 2. **深度学习工程实践**
- 数值稳定性调试技巧
- 训练/推理模式差异处理
- 梯度流分析和优化

### 3. **NLP全流程开发**
- 数据预处理和分词器训练
- 序列到序列模型设计
- 集束搜索解码算法

### 4. **PyTorch高级用法**
- 自定义模块开发
- 混合精度训练
- 分布式训练基础

## 🐛 常见问题

### Q1: 训练时Loss不下降
- 检查Embedding层是否乘以√d_model
- 验证LayerNorm初始化是否正确
- 确认掩码逻辑是否正确

### Q2: 推理时总是输出`<eos>`
- 检查T=1时刻的数值稳定性
- 验证模型是否在train/eval模式下行为一致
- 检查解码器的掩码生成

### Q3: GPU内存不足
- 减小batch_size
- 使用梯度累积
- 启用混合精度训练

### Q4: 翻译质量不佳
- 增加训练数据量
- 调整beam_size参数
- 添加长度惩罚

## 📚 学习资源

### 必读论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原始论文
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Transformer代码解读

### 参考实现
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PyTorch官方Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)

### 进阶学习
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [GPT系列论文](https://openai.com/research/gpt-4)

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出改进建议：

1. **Fork本仓库**
2. **创建功能分支** (`git checkout -b feature/AmazingFeature`)
3. **提交更改** (`git commit -m 'Add some AmazingFeature'`)
4. **推送到分支** (`git push origin feature/AmazingFeature`)
5. **开启Pull Request**

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- 感谢HuggingFace提供的`datasets`和`tokenizers`库
- 感谢PyTorch团队提供优秀的深度学习框架
- 感谢所有Transformer相关论文的作者

## 📞 联系

如有问题或建议，请通过以下方式联系：
- **GitHub Issues**: [提交问题](https://github.com/konodiodaaaaa1/PyTorch-Transformer-From-Scratch/issues)
- **Email**: your.email@example.com

---

**温馨提示**: 本项目适合有一定深度学习基础的学习者。如果你是初学者，建议先学习PyTorch基础和NLP基础知识。每个文件都包含详细注释，请结合代码和笔记PDF一起学习。

**学习建议**:
1. 先运行`debug_plus.py`理解模型结构
2. 阅读`model_components.py`中的每个组件
3. 尝试修改超参数观察效果
4. 在自己的数据集上尝试应用
