# detection_agent.py
from langchain.agents import initialize_agent, AgentType
from langchain.schema import AgentFinish
from common import create_detection_tools, llm, chat_prompt_template
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

detection_tools = create_detection_tools()
agent = initialize_agent(
    tools=detection_tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

def run_agent(query: str) -> dict:
    """
    1. 把用户 query（可能包含 base64）直接喂给 Agent
    2. 无论成功、工具报错、还是 Agent 直接回复，都返回统一的 JSON：
       {
           "think": "...",               # Agent 的思考过程（可为空）
           "result": { ... }             # 工具返回的 dict（可能含 "error"）
       }
    """
    prompt = chat_prompt_template.format_messages(
        role="目标检测",
        domain="使用工具检测图像中的目标，支持的类别从工具中检查",
        question=query
    )

    try:
        resp = agent.invoke(prompt)                 # <<< 这里返回 dict
        logger.info(f"Agent raw response: {resp}")

        # 1. 如果是 AgentFinish（直接回复） → 把它包装成 result
        if isinstance(resp.get("output"), AgentFinish):
            finish: AgentFinish = resp["output"]
            return {
                "think": "",
                "result": {"text": finish.return_values.get("output", "")}
            }

        # 2. 正常情况：resp["output"] 是字符串（可能是 JSON、也可能是纯文字）
        raw_output = resp.get("output", "")

        # 3. 尝试解析 JSON（工具返回的都是 JSON）
        try:
            parsed = json.loads(raw_output)
            return {"think": "", "result": parsed}
        except json.JSONDecodeError:
            # 不是 JSON → 直接当作文字回复
            return {"think": "", "result": {"text": raw_output}}

    except Exception as e:
        logger.exception("Agent 执行异常")
        return {"think": "", "result": {"error": str(e)}}