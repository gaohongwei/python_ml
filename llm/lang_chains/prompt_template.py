# prompt_templates.py

# 战略目标提取相关模板
# 多轮链条的 prompt 配置
strategy_templates = [
    {
        "input_key": "text",
        "output_key": "extracted_goals",
        "template": "你知道对话历史如下: {history}\n\n请从以下文本中提取出关于该公司2025年战略的主要目标和计划: {text}"
    },
    {
        "input_key": "extracted_goals",
        "output_key": "goals_analysis",
        "template": "你知道对话历史如下: {history}\n\n基于以下战略目标，分析每个目标实现的可行性与潜在风险: {extracted_goals}"
    },
    {
        "input_key": "goals_analysis",
        "output_key": "recommendations",
        "template": "你知道对话历史如下: {history}\n\n根据以下目标分析，结合行业背景和政策，给出改进建议: {goals_analysis}"
    },
    {
        "input_key": "recommendations",
        "output_key": "final_report",
        "template": "你知道对话历史如下: {history}\n\n综合以下分析，提供一个关于该公司战略目标实施的最终评估报告: {recommendations}"
    }
]

# 可以再加入其他任务模板
qa_templates = [
    {
        "input_key": "question",
        "output_key": "answer",
        "template": "请回答以下问题: {question}"
    },
    # 其他 Q&A 模板...
]
