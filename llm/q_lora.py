from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel, PeftConfig
from datasets import load_dataset
import torch

# PEFT（Parameter-Efficient Fine-Tuning）,参数高效微调
# 支持如 LoRA、QLoRA 等方法，大幅降低训练资源需求。
# QLoRA（Quantized LoRA）是 LoRA 的一种变体，结合了量化技术以进一步减少内存占用。

# === Step 1: 载入千问（Qwen）7B模型及其分词器 ===
base_model_name = "Qwen/Qwen1.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)  # 加载分词器

# 加载 Qwen 模型，并启用 4-bit 量化以减少内存消耗
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    trust_remote_code=True,  # 启用从远程加载代码，支持Qwen的自定义代码
    device_map="cpu"  # auto 自动选择设备（比如GPU）
    # 启用4-bit量化（显著减少内存占用）
    # load_in_4bit=True,       
)

# === Step 2: 准备 QLoRA 训练（冻结主模型，仅训练 LoRA Adapter） ===
# QLoRA 训练步骤：我们冻结基础模型的权重，只对 LoRA adapter 层进行训练
model = prepare_model_for_kbit_training(model)

# 配置 LoRA 相关的参数，这些参数会影响 adapter 的结构与训练效果
lora_config = LoraConfig(
    r=8,                      # LoRA矩阵的秩（低秩近似）
    lora_alpha=32,            # LoRA 缩放因子（控制 LoRA 插件的权重大小）
    target_modules=["c_attn", "q_proj", "v_proj"],  # 选择哪些模块插入 LoRA adapter（Qwen中的某些层）
    lora_dropout=0.05,        # LoRA adapter的dropout率
    bias="none",              # 不为 adapter 层的偏置项添加参数
    task_type="CAUSAL_LM"     # 设定任务类型为因果语言建模（Causal Language Modeling）
)

# 将 LoRA 配置应用到模型，得到一个带有 LoRA adapter 的模型
model = get_peft_model(model, lora_config)

# === Step 3: 加载医学问答数据集 pubmed_qa（来自 Hugging Face） ===
# 这里选择了 pubmed_qa 数据集，它包含医学领域的问题和答案，适合用于问答任务
# 这里只取前1000条数据用于测试
dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train[:1000]")  


# 数据预处理：我们将问题和背景拼接成一个输入文本，作为 QLoRA 模型的输入
def preprocess(example):
    # 拼接问题和背景信息，形成模型输入格式
    input_text = f"问题：{example['question']}\n背景：{example['context']}\n请回答："
    return tokenizer(input_text, padding="max_length", truncation=True, max_length=512)

# 对数据集进行批量处理
tokenized_dataset = dataset.map(preprocess)

# === Step 4: 设置训练参数 ===
# 这里使用的是 Hugging Face 的 Trainer 来进行训练
training_args = TrainingArguments(
    per_device_train_batch_size=2,     # 每个设备上的训练批次大小（根据显存大小可以调节）
    gradient_accumulation_steps=4,     # 梯度累积步数（用于大批次训练）
    warmup_steps=5,                   # 学习率预热步骤数
    max_steps=30,                     # 训练步数（为了演示，这里设置为30步，实际情况可以更长）
    learning_rate=2e-4,               # 学习率
    fp16=True,                         # 启用半精度训练，减少显存消耗
    logging_steps=5,                   # 日志记录频率
    output_dir="./qwen_qlora_medical", # 保存训练结果的目录
    report_to="none"                   # 不上传训练日志到任何平台
)

# === Step 5: 启动 Trainer 进行 QLoRA 微调 ===
# 使用 Trainer 类来进行模型的微调训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,  # 训练数据集
    tokenizer=tokenizer                # 分词器
)

# 开始训练
trainer.train()

# === Step 6: 保存训练得到的 LoRA Adapter 和 Tokenizer ===
# 训练完成后，保存 LoRA adapter 权重和分词器
model.save_pretrained("./qwen_med_qlora_adapter")  # 保存模型
tokenizer.save_pretrained("./qwen_med_qlora_adapter")  # 保存分词器
print("✅ LoRA adapter 已保存到 ./qwen_med_qlora_adapter")

# === Step 7: 加载保存的 Adapter 进行推理 ===
# 重新加载之前保存的 LoRA adapter 权重，并进行推理
adapter_path = "./qwen_med_qlora_adapter"
config = PeftConfig.from_pretrained(adapter_path)

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    trust_remote_code=True,  # 启用自定义代码支持
    load_in_4bit=True,       # 4-bit量化
    device_map="auto"        # 自动选择设备
)

# 载入 LoRA adapter 到模型
model = PeftModel.from_pretrained(base_model, adapter_path)

# 示例推理：提出一个医学相关的问题，模型会生成答案
input_text = "问题：什么是高血压？\n背景：高血压是一种常见的慢性病，患者血压持续升高。\n请回答："
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)  # 生成最多100个新token
print("🧠 模型输出：")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))  # 解码并输出结果

# === Step 8: 合并 adapter 生成部署版模型（可转为 ONNX、GGUF 等格式） ===
# 如果你准备部署模型，可以将 LoRA adapter 和基础模型合并为一个完整的模型
merged_model = model.merge_and_unload()  # 合并 LoRA adapter 到基础模型
merged_model.save_pretrained("./qwen_med_merged_model")  # 保存合并后的模型
tokenizer.save_pretrained("./qwen_med_merged_model")    # 保存分词器
print("📦 部署版模型已保存到 ./qwen_med_merged_model")
