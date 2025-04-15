# LLM concepts

## transformers,Hugging Face

- Python 库（toolkit / framework）
- AutoModel, AutoTokenizer 统一加载模型和 tokenizer
- Trainer 模型训练框架，类似 sklearn 的 .fit()
- datasets 配套的训练/测试数据加载工具（需 datasets 库）
- pipeline 开箱即用的推理封装，适合快速测试任务

## sample code

```
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 指定预训练模型名称（可替换为其他支持的模型）
###  小型中文模型
model_name = "Qwen/Qwen1.5-0.5B"
model_name =  "microsoft/Phi-3-mini-4k-instruct"
#从本地加载
model_name = "./models/qwen-0.5b"

# 加载因果语言模型（用于文本生成任务）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",           # 自动将模型加载到 GPU
    torch_dtype="auto",          # 自动选择最合适的数据类型（如 float16）
    trust_remote_code=True       # 允许加载远程模型中自定义的代码
)

# 加载对应的分词器（Tokenizer）
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 构建一个文本生成的 pipeline
pipe = pipeline(
    task="text-generation",      # 指定任务类型为“文本生成”
    model=model,                 # 使用上面加载的模型
    tokenizer=tokenizer,         # 使用相应的 tokenizer
    return_full_text=True,       # 返回完整文本（包括输入提示）
    max_new_tokens=500,          # 最多生成 500 个新 token
    do_sample=True               # 启用采样（生成内容更随机）
)

# 示例调用（可选）
# output = pipe("请写一个关于人工智能的故事")[0]['generated_text']
# print(output)

```

## Chapter 2. Tokens and Embeddings

### workflow

text -> token_ids->LLM->token_ids->text

### tokens

    每个 token ID 是一个唯一的编号，对应某一个token。
    这个 token 可以是：一个字符、一个单词，或者一个词的一部分（比如词根、前缀、后缀）。
    这些 ID 指向 tokenizer 内部的一张词表（vocabulary），这张词表记录了所有它“认识”的 token。

### 决定 Tokenizer 分词方式的三大关键因素

- 选择分词算法（模型设计阶段）
  - BPE（Byte Pair Encoding，字节对编码）
  - WordPiece
  - BERT, 基于 Transformer 编码器架构 的语言理解模型,Bidirectional,同时向左右看
- 分词器设计细节
- 训练语料的影响

### attention_mask

### attention_mask 是用于指示模型在计算自注意力时，哪些位置需要被关注（即哪些 token 是有效的，哪些应该被忽略）的一种标记。在很多 NLP 模型中，尤其是在 Transformer 架构中，attention mask 是一个非常重要的概念。

🌟 详细解释：
attention_mask 是一个与 input_ids 长度相同的二值序列（通常是一个张量），其值表示哪些 token 是有效的，哪些 token 是填充（padding）的位置。

1 表示该位置是一个有效的 token，模型应该考虑该 token。

0 表示该位置是填充位置，模型应该忽略该位置。

为什么需要 attention_mask？
在处理变长文本时，通常我们会对较短的文本进行 填充，使其长度与最长的文本相同。填充的部分并没有实际的语义内容，但在计算自注意力（self-attention）时，模型应该忽略这些填充部分的影响。attention_mask 就是用来标记这些填充位置的。

### Token Embeddings

### Text Embeddings

## Chapter 3. Looking Inside Large Language Models

### Sample code

from transformers import AutoModelForCausalLM, AutoTokenizer

### 小型中文模型

```
model_name = "Qwen/Qwen1.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name,
    device_map="cpu",# auto
    torch_dtype="auto",
    trust_remote_code=True,
)

# Create a pipeline

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text= False ,
    max_new_tokens=50,
    do_sample= False,
)
```

The model does not generate the text all in one operation;
it actually generates one token at a time.
Each token generation step is one forward pass through the model
After each token generation, we tweak the input prompt for the next generation step by appending the output token to the end of the input prompt.
Software around the neural network basically runs it in a loop to sequentially expand the generated text until completion.

#### The Components of the Forward Pass

In addition to the loop, two key internal components are the tokenizer and the language modeling head (LM head).

## Chapter 12. Fine-Tuning Generation Models
