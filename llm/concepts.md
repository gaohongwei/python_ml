### transformers,Hugging Face

- Python 库（toolkit / framework）
- AutoModel, AutoTokenizer 统一加载模型和 tokenizer 的接
- Trainer 模型训练框架，类似 sklearn 的 .fit()
- datasets 配套的训练/测试数据加载工具（需 datasets 库）
- pipeline 开箱即用的推理封装，适合快速测试任务
- Common usage:

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

### datasets,Hugging Face

- list_datasets
  > > > from datasets import list_datasets
  > > > all_datasets = list_datasets()
  > > > <stdin>:1: FutureWarning: list_datasets is deprecated and will be removed in the next major version of datasets. Use 'huggingface_hub.list_datasets' instead.

from transformer import
AutoModelForCausalLLM,
AutoTokenizer,
pipeline

model = AutoModelForCausalLLM.from_pretrained()
tokenizer = AutoTokenizer.from_pretrained()

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.<|assistant|>"

input_ids = tokenizer(prompt,return_tensors="pt").input_ids.to("cpu")

generation_output = model.generate(
input_ids=input_ids,
max_new_tokens=20
)

## concept

### attention_mask 是用于指示模型在计算自注意力时，哪些位置需要被关注（即哪些 token 是有效的，哪些应该被忽略）的一种标记。在很多 NLP 模型中，尤其是在 Transformer 架构中，attention mask 是一个非常重要的概念。

🌟 详细解释：
attention_mask 是一个与 input_ids 长度相同的二值序列（通常是一个张量），其值表示哪些 token 是有效的，哪些 token 是填充（padding）的位置。

1 表示该位置是一个有效的 token，模型应该考虑该 token。

0 表示该位置是填充位置，模型应该忽略该位置。

为什么需要 attention_mask？
在处理变长文本时，通常我们会对较短的文本进行 填充，使其长度与最长的文本相同。填充的部分并没有实际的语义内容，但在计算自注意力（self-attention）时，模型应该忽略这些填充部分的影响。attention_mask 就是用来标记这些填充位置的。
