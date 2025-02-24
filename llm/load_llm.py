
from transformers import AutoModelForCausalLM, AutoTokenizer

###  小型中文模型
model_name = "Qwen/Qwen1.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, 
    device_map="cpu",# auto
    torch_dtype="auto", 
    trust_remote_code=True,
)


# 根据模型名称自动选择并加载对应的分词器（Tokenizer）
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

prompt = "Write an email apologizing to Sarah for the tragicgardening mishap. Explain how it happened.<|assistant|>"

input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
input_ids = tokenizer(prompt,return_tensors="pt").to("cpu")
>>> input_ids
tensor([[14350,   385,  4876, 27746,  5281,   304, 19235,   363,   278, 25305,
           293, 29887,  7879,   292,   286,   728,   481, 29889, 12027,  7420,
           920,   372,  9559, 29889, 32001]])
# Generate the text
generation_output = model.generate(
    input_ids=input_ids,
    max_new_tokens=20,
    use_cache=False  # <-- 临时绕过问题
)
outputs = model.generate(**inputs, max_new_tokens=100)

# Print the output
>>> generation_output
tensor([[14350,   385,  4876, 27746,  5281,   304, 19235,   363,   278, 25305,
           293, 29887,  7879,   292,   286,   728,   481, 29889, 12027,  7420,
           920,   372,  9559, 29889, 32001,  3323,   622, 29901,   317,  3742,
           406,  6225, 11763,   363,   278, 19906,   292,   341,   728,   481,
            13,    13,    13, 29928,   799]])


print(tokenizer.decode(outputs[0], skip_special_tokens=True))
