from agent.tools import TOOL_METADATA, TOOLS


DEFAULT_CONTEXT = {
    "user_role": "student",
    "page": "general"
}


INTENT_KEYWORDS = {
    "teacher_schedule": ["老师", "有空", "时间", "约", "会议", "周一", "周二", "周三", "周四", "周五"],
    "course_info": ["课程", "这门课", "在哪里上", "教室", "谁教", "Agent开发入门", "Python基础"],
    "knowledge_base": ["制度", "规则", "请假", "考勤", "作业", "生病", "资料", "怎么办"],
    "agent_learning": ["Agent是什么", "agent是什么", "Agent loop", "工具调用", "RAG", "rag", "embedding", "向量", "切块", "chunk"]
}


def detect_intents(user_input: str) -> list:
    intents = []

    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in user_input for keyword in keywords):
            intents.append(intent)

    return intents


def normalize_context(context: dict | None) -> dict:
    normalized = DEFAULT_CONTEXT.copy()

    if context:
        normalized.update(context)

    return normalized


def filter_tools_by_permission(tool_names: list, user_role: str) -> list:
    return [
        tool_name for tool_name in tool_names
        if user_role in TOOL_METADATA[tool_name]["allowed_roles"]
    ]


def filter_tools_by_page(tool_names: list, page: str) -> list:
    return [
        tool_name for tool_name in tool_names
        if page in TOOL_METADATA[tool_name]["pages"]
    ]


def filter_tools_by_intent(tool_names: list, intents: list) -> list:
    if not intents:
        return tool_names

    matched_tools = [
        tool_name for tool_name in tool_names
        if any(intent in TOOL_METADATA[tool_name]["intents"] for intent in intents)
    ]

    return matched_tools or tool_names


def select_tool_names(user_input: str, context: dict | None = None) -> list:
    normalized_context = normalize_context(context)
    tool_names = list(TOOL_METADATA.keys())
    tool_names = filter_tools_by_permission(tool_names, normalized_context["user_role"])
    tool_names = filter_tools_by_page(tool_names, normalized_context["page"])
    tool_names = filter_tools_by_intent(tool_names, detect_intents(user_input))

    return tool_names


def select_tools(user_input: str, context: dict | None = None) -> list:
    tool_names = select_tool_names(user_input, context)
    selected_tools = [
        TOOL_METADATA[tool_name]["schema"]
        for tool_name in tool_names
    ]

    return selected_tools or TOOLS
