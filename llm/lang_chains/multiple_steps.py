from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

from prompt_templates import strategy_templates, qa_templates

# 初始化语言模型
llm = OpenAI(temperature=0.7)

# 配置使用的记忆类型：'buffer' 或 'summary'
memory_type = "buffer"  # ← 更改为 "summary" 可切换为总结型记忆

# 根据配置初始化记忆模块
if memory_type == "buffer":
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
elif memory_type == "summary":
    memory = ConversationSummaryMemory(llm=llm, memory_key="history", return_messages=True)
else:
    raise ValueError("Unsupported memory type. Choose 'buffer' or 'summary'.")


# 构建链条列表
chains = []
for config in strategy_templates:
    prompt = PromptTemplate(
        input_variables=["history", config["input_key"]],
        template=config["template"]
    )
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    chains.append({
        "chain": chain,
        "input_key": config["input_key"],
        "output_key": config["output_key"]
    })

# 初始输入内容
context = {"text": "公司2025年战略报告文本..."}

# 顺序执行链条，依次传递上下文
for step in chains:
    input_val = context[step["input_key"]]
    output_val = step["chain"].run({step["input_key"]: input_val})
    context[step["output_key"]] = output_val

# 输出最终结果
print(context["final_report"])
