from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.tools import Calculator, SearchResults
from langchain.tools import DuckDuckGoSearchResults, Calculator

# 初始化 LLM
llm = OpenAI(temperature=0.7)

# 定义工具：计算器和搜索引擎
tools = [
    Tool(
        name="Calculator",
        func=Calculator().run,  # 假设 Calculator 类有一个 .run() 方法来执行计算
        description="用于执行数学运算"
    ),
    Tool(
        name="SearchEngine",
        func=SearchResults().run,  # 假设 SearchResults 类有一个 .run() 方法来执行搜索
        description="用于执行网页搜索"
    )
]

# 初始化 Agent
agent = initialize_agent(
    tools, 
    llm, 
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # 使用 ReAct 框架
    verbose=True
)

# 创建示例查询：首先是数学计算任务
math_query = "What is the result of 123 + 456?"

# 使用 Agent 执行查询
math_result = agent.run(math_query)
print("数学计算结果:", math_result)

# 创建另一个查询：搜索任务
search_query = "What is the current price of MacBook Pro?"

# 使用 Agent 执行查询
search_result = agent.run(search_query)
print("MacBook Pro 当前价格:", search_result)
