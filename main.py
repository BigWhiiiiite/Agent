import json
import math
import os
from pathlib import Path

from openai import OpenAI


MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
DOCS_DIR = Path(__file__).parent / "docs"
EMBEDDING_CACHE_PATH = Path(__file__).parent / "embedding_cache.json"
VECTOR_INDEX_PATH = Path(__file__).parent / "vector_index.json"
TOP_K = 3
MAX_AGENT_STEPS = 6
MIN_SEMANTIC_SIMILARITY = float(os.getenv("MIN_SEMANTIC_SIMILARITY", "0.2"))


def load_embedding_cache() -> dict:
    if not EMBEDDING_CACHE_PATH.exists():
        return {}

    return json.loads(EMBEDDING_CACHE_PATH.read_text(encoding="utf-8"))


EMBEDDING_CACHE = load_embedding_cache()


def save_embedding_cache() -> None:
    EMBEDDING_CACHE_PATH.write_text(
        json.dumps(EMBEDDING_CACHE, ensure_ascii=False),
        encoding="utf-8"
    )


def load_vector_index() -> dict:
    if not VECTOR_INDEX_PATH.exists():
        return {}

    return json.loads(VECTOR_INDEX_PATH.read_text(encoding="utf-8"))


def save_vector_index(index: dict) -> None:
    VECTOR_INDEX_PATH.write_text(
        json.dumps(index, ensure_ascii=False),
        encoding="utf-8"
    )


def build_tool_success(tool_name: str, data: dict) -> dict:
    return {
        "ok": True,
        "tool_name": tool_name,
        "data": data
    }


def build_tool_error(tool_name: str | None, error_type: str, message: str) -> dict:
    return {
        "ok": False,
        "tool_name": tool_name,
        "error": {
            "type": error_type,
            "message": message
        }
    }


def build_tool_message(tool_call_id: str, payload: dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(payload, ensure_ascii=False)
    }


def parse_tool_arguments(arguments) -> dict:
    if isinstance(arguments, str):
        return json.loads(arguments)

    if isinstance(arguments, dict):
        return arguments

    raise TypeError("工具参数必须是 JSON 字符串或 dict。")


def query_teacher_schedule(teacher_name: str, date: str) -> dict:
    fake_db = {
        ("李老师", "周五"): ["14:00", "15:00", "16:30"],
        ("王老师", "周三"): ["10:00", "11:00"],
    }

    slots = fake_db.get((teacher_name, date), [])
    return {
        "teacher_name": teacher_name,
        "date": date,
        "available_slots": slots
    }


def query_course_info(course_name: str) -> dict:
    fake_db = {
        "Agent开发入门": {
            "teacher": "李老师",
            "time": "周五 14:00",
            "classroom": "A101",
            "description": "从工具调用、Agent loop 和 RAG 的基础概念开始，带你做一个最小 Agent。"
        },
        "Python基础": {
            "teacher": "王老师",
            "time": "周三 10:00",
            "classroom": "B203",
            "description": "学习 Python 语法、函数、字典、列表和简单项目实践。"
        }
    }

    course_info = fake_db.get(course_name)

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


def build_search_terms(query: str) -> set:
    cleaned_query = query
    for char in " ，。？！?：:、\n\t":
        cleaned_query = cleaned_query.replace(char, "")

    terms = set()

    if cleaned_query:
        terms.add(cleaned_query)

    for index in range(len(cleaned_query) - 1):
        terms.add(cleaned_query[index:index + 2])

    return terms


def load_knowledge_chunks() -> list:
    chunks = []

    for path in DOCS_DIR.glob("*.txt"):
        content = path.read_text(encoding="utf-8")
        paragraphs = [
            paragraph.strip()
            for paragraph in content.split("\n\n")
            if paragraph.strip()
        ]

        for index, paragraph in enumerate(paragraphs, start=1):
            chunks.append({
                "source": path.name,
                "chunk_id": f"{path.stem}-{index}",
                "content": paragraph
            })

    return chunks


def score_chunk(query: str, terms: set, chunk: dict) -> int:
    content = chunk["content"]
    score = sum(1 for term in terms if term in content)

    if query in content:
        score += 5

    return score


def search_knowledge_base(query: str) -> dict:
    terms = build_search_terms(query)
    results = []

    for chunk in load_knowledge_chunks():
        score = score_chunk(query, terms, chunk)

        if score > 0:
            results.append({
                **chunk,
                "score": score
            })

    results.sort(key=lambda item: item["score"], reverse=True)
    top_results = results[:TOP_K]

    return {
        "query": query,
        "found": bool(top_results),
        "result_count": len(top_results),
        "top_k": TOP_K,
        "results": top_results
    }


def get_embedding(text: str) -> list:
    if text in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[text]

    client = OpenAI()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        encoding_format="float"
    )
    embedding = response.data[0].embedding
    EMBEDDING_CACHE[text] = embedding
    save_embedding_cache()
    return embedding


def cosine_similarity(vector_a: list, vector_b: list) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def semantic_search_knowledge_base(query: str) -> dict:
    query_embedding = get_embedding(query)
    vector_index = get_vector_index()
    results = []

    for chunk in vector_index["chunks"]:
        chunk_embedding = chunk["embedding"]
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        results.append({
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"],
            "similarity": round(similarity, 4)
        })

    results.sort(key=lambda item: item["similarity"], reverse=True)
    top_results = [
        result for result in results
        if result["similarity"] >= MIN_SEMANTIC_SIMILARITY
    ][:TOP_K]

    return {
        "query": query,
        "found": bool(top_results),
        "result_count": len(top_results),
        "embedding_model": EMBEDDING_MODEL,
        "index_path": VECTOR_INDEX_PATH.name,
        "min_similarity": MIN_SEMANTIC_SIMILARITY,
        "top_k": TOP_K,
        "results": top_results
    }


def build_vector_index() -> dict:
    chunks = []

    for chunk in load_knowledge_chunks():
        chunks.append({
            **chunk,
            "embedding": get_embedding(chunk["content"])
        })

    index = {
        "embedding_model": EMBEDDING_MODEL,
        "chunks": chunks
    }
    save_vector_index(index)
    return index


def get_vector_index() -> dict:
    index = load_vector_index()
    current_chunks = load_knowledge_chunks()
    indexed_chunks = index.get("chunks", [])

    current_signature = [
        (chunk["source"], chunk["chunk_id"], chunk["content"])
        for chunk in current_chunks
    ]
    indexed_signature = [
        (chunk["source"], chunk["chunk_id"], chunk["content"])
        for chunk in indexed_chunks
    ]

    if (
        index.get("embedding_model") != EMBEDDING_MODEL
        or current_signature != indexed_signature
    ):
        return build_vector_index()

    return index


TOOL_SCHEMA = {
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
    TOOL_SCHEMA,
    COURSE_TOOL_SCHEMA,
    KNOWLEDGE_BASE_TOOL_SCHEMA,
    SEMANTIC_KNOWLEDGE_BASE_TOOL_SCHEMA
]


TOOL_REGISTRY = {
    "query_teacher_schedule": query_teacher_schedule,
    "query_course_info": query_course_info,
    "search_knowledge_base": search_knowledge_base,
    "semantic_search_knowledge_base": semantic_search_knowledge_base
}


def execute_tool_call(tool_call: dict) -> dict:
    tool_call_id = tool_call.get("id", "unknown_tool_call")
    function_info = tool_call.get("function", {})
    function_name = function_info.get("name")
    raw_arguments = function_info.get("arguments", {})

    if function_name not in TOOL_REGISTRY:
        payload = build_tool_error(
            function_name,
            "unknown_tool",
            f"工具 {function_name} 没有注册，无法执行。"
        )
        return build_tool_message(tool_call_id, payload)

    try:
        arguments = parse_tool_arguments(raw_arguments)
    except (json.JSONDecodeError, TypeError) as error:
        payload = build_tool_error(
            function_name,
            "invalid_arguments",
            f"工具参数解析失败：{error}"
        )
        return build_tool_message(tool_call_id, payload)

    try:
        tool_function = TOOL_REGISTRY[function_name]
        result = tool_function(**arguments)
        payload = build_tool_success(function_name, result)
    except TypeError as error:
        payload = build_tool_error(
            function_name,
            "argument_mismatch",
            f"工具参数和函数签名不匹配：{error}"
        )
    except Exception as error:
        payload = build_tool_error(
            function_name,
            "tool_runtime_error",
            f"工具执行失败：{error}"
        )

    return build_tool_message(tool_call_id, payload)


def call_llm(messages: list, tools: list) -> dict:
    client = OpenAI()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    assistant_message = response.choices[0].message
    return assistant_message.model_dump(exclude_none=True)


def print_messages(title: str, messages: list) -> None:
    print(f"\n{title}")
    for index, message in enumerate(messages, start=1):
        print(f"{index}. role={message['role']}")
        print(json.dumps(message, ensure_ascii=False, indent=2))


def print_tool_trace(tool_call: dict, tool_message: dict) -> None:
    function_info = tool_call.get("function", {})
    function_name = function_info.get("name")
    raw_arguments = function_info.get("arguments", {})
    payload = json.loads(tool_message["content"])
    status = "成功" if payload["ok"] else "失败"

    print("\n工具调用 trace")
    print(f"tool={function_name}")
    print(f"arguments={raw_arguments}")
    print(f"status={status}")

    if not payload["ok"]:
        print(f"error={payload['error']['message']}")


def create_initial_messages() -> list:
    return [
        {
            "role": "system",
            "content": (
                "你是一个校园助手。"
                "你可以查询老师空闲时间、课程信息，也可以搜索本地知识库。"
                "如果用户的问题需要工具，并且参数已经足够，就调用最合适的工具。"
                "回答学校制度、课程规则、Agent或RAG概念时，优先搜索本地知识库。"
                "如果用户的问题和知识库里的原文措辞可能不一致，优先使用语义检索工具。"
                "如果缺少必要参数，就先追问用户，不要编造参数。"
                "回答知识库问题时，要基于工具返回的 results，并可简短提及来源 source。"
                "如果知识库工具返回 found=false，就如实说明没有查到相关资料。"
                "工具结果里的 ok=false 表示工具失败，需要根据 error.message 向用户简短说明。"
                "最终回答要简短、自然。"
            )
        }
    ]


def run_agent(messages: list, user_input: str, debug: bool = False) -> str:
    messages.append({
        "role": "user",
        "content": user_input
    })

    if debug:
        print_messages("用户输入后 messages", messages)

    for _ in range(MAX_AGENT_STEPS):
        assistant_message = call_llm(messages, tools=TOOLS)
        messages.append(assistant_message)

        if debug:
            print_messages("模型返回 assistant message 后", messages)

        if "tool_calls" not in assistant_message:
            return assistant_message["content"]

        for tool_call in assistant_message["tool_calls"]:
            tool_message = execute_tool_call(tool_call)
            messages.append(tool_message)

            if debug:
                print_tool_trace(tool_call, tool_message)

        if debug:
            print_messages("工具执行完成并加入 tool message 后", messages)

    return "工具调用轮数超过上限，已停止。请换一种问法或稍后再试。"


if __name__ == "__main__":
    messages = create_initial_messages()

    while True:
        user_input = input("\n请输入你的问题，输入 exit 退出：")

        if user_input == "exit":
            print("已退出。")
            break

        answer = run_agent(messages, user_input, debug=True)
        print("\n最终回答：")
        print(answer)
