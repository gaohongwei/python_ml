from langchain.llms import OpenAI
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 定义模型选择函数
def get_llm(model_type: str, model_path: str = None):
    """
    根据模型类型加载相应的 LLM 模型。
    :param model_type: 模型类型 ('openai', 'llama', 'huggingface')
    :param model_path: 模型路径（适用于本地模型，如 LlamaCpp）
    :return: 返回一个 LLM 模型实例
    """
    if model_type == 'openai':
        # 使用 OpenAI 的模型
        return OpenAI(model="text-davinci-003")
    elif model_type == 'llama' and model_path:
        # 使用 LlamaCpp 的模型（需要提供路径）
        return LlamaCpp(model_path=model_path)
    elif model_type == 'huggingface':
        # 使用 HuggingFace 模型（可以是任何已发布的模型）
        from langchain.llms import HuggingFaceLLM
        return HuggingFaceLLM(model="distilgpt2")  # 例子模型，你可以替换成任何 HuggingFace 模型
    else:
        raise ValueError("Unsupported model type or missing model path")

# 使用模型类型来加载不同的 LLM 模型
model_type = 'llama'  # 可以选择 'openai', 'llama', 'huggingface'
model_path = "path_to_your_llama_model"  

# 获取 LLM 实例
llm = get_llm(model_type=model_type, model_path=model_path)

# 创建一个简单的提示模板
prompt_template = "What is the capital of {country}?"

# 使用模板创建一个 PromptTemplate 实例
prompt = PromptTemplate(input_variables=["country"], template=prompt_template)

# 使用 LLMChain 连接模板和 LLM
llm_chain = LLMChain(prompt=prompt, llm=llm)

# 提供输入并运行链
response = llm_chain.run({"country": "France"})

# 输出模型的回答
print(f"Answer: {response}")
