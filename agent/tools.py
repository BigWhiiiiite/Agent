from agent.data_store import load_courses, load_teacher_schedule
from agent.rag import search_knowledge_base, semantic_search_knowledge_base


def query_teacher_schedule(teacher_name: str, date: str) -> dict:
    schedules = load_teacher_schedule()
    slots = []

    for schedule in schedules:
        if (
            schedule["teacher_name"] == teacher_name
            and schedule["date"] == date
        ):
            slots = schedule["available_slots"]
            break

    return {
        "teacher_name": teacher_name,
        "date": date,
        "available_slots": slots
    }


def query_course_info(course_name: str) -> dict:
    courses = load_courses()
    course_info = courses.get(course_name)

    if course_info is None:
        return {
            "course_name": course_name,
            "found": False,
            "message": "没有查到这门课的信息。"
        }

    return {
        "course_name": course_name,
        "found": True,
        **course_info
    }


TEACHER_SCHEDULE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_teacher_schedule",
        "description": "查询某位老师在指定日期的空闲时间段，用于安排会议或约时间。",
        "parameters": {
            "type": "object",
            "properties": {
                "teacher_name": {
                    "type": "string",
                    "description": "老师姓名，比如李老师"
                },
                "date": {
                    "type": "string",
                    "description": "日期，比如周五"
                }
            },
            "required": ["teacher_name", "date"]
        }
    }
}


COURSE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_course_info",
        "description": "查询课程介绍、授课老师、上课时间和教室。",
        "parameters": {
            "type": "object",
            "properties": {
                "course_name": {
                    "type": "string",
                    "description": "课程名称，比如Agent开发入门或Python基础"
                }
            },
            "required": ["course_name"]
        }
    }
}


KNOWLEDGE_BASE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "搜索本地知识库，用于回答学校制度、课程规则、Agent和RAG概念说明等资料型问题。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要搜索的问题或关键词，比如请假制度、RAG是什么"
                }
            },
            "required": ["query"]
        }
    }
}


SEMANTIC_KNOWLEDGE_BASE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "semantic_search_knowledge_base",
        "description": "用 embedding 语义检索本地知识库，适合用户表达和文档措辞不完全一致的资料型问题。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要语义检索的问题，比如生病不能上课怎么办、为什么要把文档切块"
                }
            },
            "required": ["query"]
        }
    }
}


TOOLS = [
    TEACHER_SCHEDULE_TOOL_SCHEMA,
    COURSE_TOOL_SCHEMA,
    KNOWLEDGE_BASE_TOOL_SCHEMA,
    SEMANTIC_KNOWLEDGE_BASE_TOOL_SCHEMA
]

TOOL_METADATA = {
    "query_teacher_schedule": {
        "schema": TEACHER_SCHEDULE_TOOL_SCHEMA,
        "intents": ["teacher_schedule"],
        "allowed_roles": ["student", "teacher", "admin"],
        "pages": ["general", "teacher"],
        "risk_level": "low"
    },
    "query_course_info": {
        "schema": COURSE_TOOL_SCHEMA,
        "intents": ["course_info"],
        "allowed_roles": ["student", "teacher", "admin"],
        "pages": ["general", "course"],
        "risk_level": "low"
    },
    "search_knowledge_base": {
        "schema": KNOWLEDGE_BASE_TOOL_SCHEMA,
        "intents": ["knowledge_base", "agent_learning"],
        "allowed_roles": ["student", "teacher", "admin"],
        "pages": ["general", "knowledge_base", "course"],
        "risk_level": "low"
    },
    "semantic_search_knowledge_base": {
        "schema": SEMANTIC_KNOWLEDGE_BASE_TOOL_SCHEMA,
        "intents": ["knowledge_base", "agent_learning"],
        "allowed_roles": ["student", "teacher", "admin"],
        "pages": ["general", "knowledge_base", "course"],
        "risk_level": "low"
    }
}


TOOL_REGISTRY = {
    "query_teacher_schedule": query_teacher_schedule,
    "query_course_info": query_course_info,
    "search_knowledge_base": search_knowledge_base,
    "semantic_search_knowledge_base": semantic_search_knowledge_base
}
