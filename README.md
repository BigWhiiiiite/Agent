# Minimal Agent Learning Project

这是一个教学式 Python 项目，用来一步一步理解 Agent 开发。

项目重点不是工程复杂度，而是看清楚一个最小 Agent 的骨架：

```text
用户输入
-> 模型判断是否需要工具
-> 程序执行工具
-> 工具结果回传 messages
-> 模型基于结果输出最终回答
```

当前项目已经包含：

- 真实 OpenAI Chat Completions 调用
- 多工具调用
- 命令行多轮对话
- 关键词 RAG
- 文档 chunk 切块
- embedding 语义检索 RAG
- 本地 vector index 缓存
- 工具错误处理
- Agent 最大轮数保护
- 调试 trace
- Tool Router 动态工具选择
- Agent Trace 审计日志
- FastAPI 多轮 HTTP 接口
- 短期 memory 裁剪，防止 messages 无限增长
- Eval 测试集

## 文件结构

```text
.
├── agent/
│   ├── __init__.py
│   ├── config.py
│   ├── core.py
│   ├── debug.py
│   ├── executor.py
│   ├── llm.py
│   ├── memory.py
│   ├── rag.py
│   ├── router.py
│   ├── session_store.py
│   ├── tracing.py
│   └── tools.py
├── main.py
├── app.py
├── eval_cases.json
├── requirements.txt
├── scripts/
│   └── run_eval.py
├── data/
│   ├── courses.json
│   └── teacher_schedule.json
└── docs/
    ├── agent_notes.txt
    ├── course_rules.txt
    └── school_rules.txt
```

`main.py` 是命令行入口。

`app.py` 是 FastAPI HTTP 服务入口。

`eval_cases.json` 是评估用例。

`scripts/run_eval.py` 会运行本地 eval，不调用 LLM，不消耗 API。

`data/` 是结构化业务数据，课程和老师时间工具会从这里读取数据。

`agent/` 是 Agent 核心代码：

```text
config.py    配置项，比如模型名、缓存路径、TOP_K
core.py      Agent loop 和初始 messages
debug.py     messages 和 tool trace 打印
executor.py  工具执行器，负责执行 tool_call 和错误包装
llm.py       OpenAI 模型调用
memory.py    短期上下文裁剪，保留 system 和最近几轮完整对话
rag.py       chunk、关键词检索、embedding、vector index
router.py    根据权限、页面场景和意图筛选候选工具
session_store.py  HTTP 会话存储，用 session_id 保存多轮 messages
tracing.py   写入 Agent 审计日志，便于复盘和评估
tools.py     业务工具、工具 schema、工具注册表
```

`docs/` 是本地知识库，RAG 工具会从这里检索资料。

程序运行过程中可能生成：

```text
embedding_cache.json
vector_index.json
logs/agent_trace.jsonl
```

这些文件是本地运行数据，已经放进 `.gitignore`，不会提交到 GitHub。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置 API Key

运行前需要设置 OpenAI API key：

```bash
export OPENAI_API_KEY="你的 API key"
```

默认模型是：

```text
gpt-5-mini
```

默认 embedding 模型是：

```text
text-embedding-3-small
```

也可以通过环境变量修改：

```bash
export OPENAI_MODEL="gpt-5-mini"
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
export MIN_SEMANTIC_SIMILARITY="0.2"
export MAX_CONTEXT_MESSAGES="24"
```

## 运行

```bash
python3 main.py
```

输入 `exit` 退出。

## 运行 HTTP API

启动服务：

```bash
uvicorn app:app --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

聊天接口：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-user-1",
    "message": "请假制度是什么？",
    "context": {
      "user_role": "student",
      "page": "general"
    }
  }'
```

返回格式：

```json
{
  "answer": "...",
  "context": {
    "user_role": "student",
    "page": "general"
  },
  "session_id": "demo-user-1",
  "message_count": 4
}
```

同一个 `session_id` 会复用同一份 `messages`，所以 HTTP API 已经可以支持多轮聊天。比如第二轮可以继续使用同一个 `session_id`：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-user-1",
    "message": "那如果请假超过三天呢？",
    "context": {
      "user_role": "student",
      "page": "general"
    }
  }'
```

如果想重新开始这个会话，可以在请求里加：

```json
{
  "session_id": "demo-user-1",
  "message": "重新问一个问题",
  "reset": true
}
```

也可以查看或删除当前内存里的会话：

```bash
curl http://127.0.0.1:8000/sessions
curl -X DELETE http://127.0.0.1:8000/sessions/demo-user-1
```

注意：当前会话存储只是教学版内存字典。服务重启后会话会消失，多进程部署时也不会共享会话。真实项目里通常会换成 Redis、数据库或平台自带的会话存储。

## 运行 Eval

```bash
python3 scripts/run_eval.py
```

第一版 eval 只评估本地确定性能力：

```text
Tool Router 是否选出预期候选工具
关键词 RAG 是否命中预期 chunk
结构化数据工具是否返回预期字段
```

它不会调用真实 LLM，也不会调用 embedding API。

示例输出：

```text
[PASS] router_teacher_schedule
[PASS] rag_leave_policy
Passed 10/10 eval cases.
```

## 当前工具

### query_teacher_schedule

查询某位老师在指定日期的空闲时间。

数据来自：

```text
data/teacher_schedule.json
```

示例问题：

```text
帮我看看李老师周五什么时候有空
```

### query_course_info

查询课程介绍、授课老师、时间和教室。

数据来自：

```text
data/courses.json
```

示例问题：

```text
Agent开发入门这门课在哪里上？
```

### search_knowledge_base

关键词检索本地知识库。

它会：

```text
读取 docs/*.txt
-> 按空行切成 chunks
-> 用关键词给 chunk 打分
-> 取前 3 个 chunks
-> 交给模型回答
```

示例问题：

```text
请假制度是什么？
```

### semantic_search_knowledge_base

embedding 语义检索本地知识库。

它会：

```text
把用户问题转成 embedding
-> 读取或构建本地 vector_index.json
-> 计算余弦相似度
-> 过滤低于 MIN_SEMANTIC_SIMILARITY 的结果
-> 取最相近的 3 个 chunks
-> 交给模型回答
```

示例问题：

```text
生病不能上课怎么办？
```

## RAG 当前实现

当前 RAG 分两种。

第一种是关键词 RAG：

```text
用户问题
-> 拆成搜索词
-> 和每个 chunk 做字面匹配
-> 按命中分数排序
```

第二种是语义 RAG：

```text
用户问题
-> embedding
-> 本地 vector index 中的 chunk embedding
-> cosine similarity
-> 按语义相似度排序
```

两种方式都会返回 `TOP_K = 3` 个 chunk。

也就是说，不是把整个知识库都发给模型，而是只把最相关的 3 个片段放回 `messages`。

## 本地 Vector Index

`vector_index.json` 是这个项目里的最小向量索引。

它保存的是：

```text
chunk source
chunk_id
chunk content
chunk embedding
```

语义检索时，不需要每次重新给所有 chunk 生成 embedding。程序会：

```text
检查 vector_index.json 是否存在
-> 检查 embedding 模型和当前 docs chunks 是否匹配
-> 如果匹配，直接复用索引
-> 如果不匹配，重新构建索引
```

这可以理解成一个非常简化的向量数据库。

真正的向量数据库会继续解决：

```text
大量向量存储
快速相似度检索
metadata 过滤
增量更新
并发访问
```

## Agent Loop

核心循环在 `run_agent()`：

```python
for _ in range(MAX_AGENT_STEPS):
    assistant_message = call_llm(messages, tools=TOOLS)
    messages.append(assistant_message)

    if "tool_calls" not in assistant_message:
        return assistant_message["content"]

    for tool_call in assistant_message["tool_calls"]:
        tool_message = execute_tool_call(tool_call)
        messages.append(tool_message)
```

这段代码表达了 Agent 的核心：

```text
模型如果直接回答，循环结束。
模型如果请求工具，程序执行工具，再把结果放回 messages。
```

当前 Agent loop 还加了最大轮数保护：

```text
MAX_AGENT_STEPS = 6
```

如果模型一直请求工具、不输出最终回答，程序会停止并返回提示，避免无限循环。

## Memory 裁剪

HTTP API 支持 `session_id` 后，同一个用户的 `messages` 会不断变长。

真实项目里这会带来几个问题：

```text
请求 token 越来越多，成本变高
上下文越长，响应越慢
超过模型上下文窗口后，请求可能失败
很早以前的历史可能干扰当前问题
```

所以当前项目加了一个教学版短期 memory：

```text
保留 system message
按完整对话轮次裁剪历史
优先保留最近几轮
默认最多保留 MAX_CONTEXT_MESSAGES = 24 条 message
```

这里特意没有简单地截取最后 N 条 message，因为 Agent 历史里可能有工具调用：

```text
assistant tool_calls
-> tool result
-> assistant final answer
```

如果只保留其中一半，模型会看到不完整的工具调用历史，可能导致 API 报错或回答混乱。

所以 `memory.py` 会按“用户一轮问题 + 后续 assistant/tool 消息”作为一个整体来保留或丢弃。

## 工具执行结果

真实应用里，模型可能会：

```text
调用不存在的工具
生成坏的 JSON 参数
漏掉必填参数
触发工具内部异常
```

所以 `execute_tool_call()` 不再直接让程序崩掉，而是把工具结果统一包装成：

```json
{
  "ok": true,
  "tool_name": "query_course_info",
  "data": {}
}
```

或者：

```json
{
  "ok": false,
  "tool_name": "query_course_info",
  "error": {
    "type": "invalid_arguments",
    "message": "工具参数解析失败"
  }
}
```

这样模型第二轮看到工具失败时，可以向用户说明问题或追问信息。

## Tool Router

真实 Agent 应用通常不会每轮都把所有工具暴露给模型。

原因是：

```text
工具太多会增加 token 成本
模型更容易选错工具
敏感工具不能随便暴露
不同页面或用户角色可用工具不同
```

所以项目里增加了 `agent/router.py`。

当前路由分三层：

```text
1. 权限过滤：根据 user_role 判断用户能不能用某类工具
2. 场景过滤：根据 page 判断当前页面适合哪些工具
3. 意图过滤：根据用户问题判断本轮更可能需要哪些工具
```

命令行入口里默认 context 是：

```python
context = {
    "user_role": "student",
    "page": "general"
}
```

当前 `detect_intents()` 使用关键词做简化意图识别。这个实现不是重点，重点是结构：

```text
用户问题 + 产品上下文
-> select_tools()
-> 候选工具列表
-> LLM 在候选工具中选择是否调用
```

后续可以把 `detect_intents()` 替换成：

```text
LLM 分类器
embedding-based tool retrieval
业务规则引擎
```

面试里可以说：

```text
权限类规则必须由程序侧硬控制，不能完全交给模型。
意图路由可以先用规则实现，后续替换为 LLM router 或语义工具检索。
```

## Agent Trace

项目会把每次用户请求写入：

```text
logs/agent_trace.jsonl
```

`.jsonl` 是一行一个 JSON，适合做日志。

每条 trace 会记录：

```text
timestamp
user_input
context
selected_tools
tool_calls
final_answer
stop_reason
```

其中 `tool_calls` 会记录：

```text
tool_name
arguments
result_summary
```

RAG 工具的结果摘要会保留：

```text
found
result_count
sources
chunk_id
```

这样当 Agent 答错时，可以复盘：

```text
Router 是否选错候选工具
LLM 是否调用了正确工具
工具参数是否正确
RAG 是否搜到正确资料
最终回答是否基于工具结果
```

## 实际应用问题和当前解法

### 模型选错工具或参数不完整

当前解法：

```text
用清晰的 tool name、description、parameters
Tool Router 先筛选候选工具，减少模型误选
在 system prompt 里要求缺参数时先追问
工具执行层返回 ok=false 错误结构
```

### 工具调用失败导致程序崩溃

当前解法：

```text
execute_tool_call 捕获参数解析错误、未知工具、参数不匹配和运行时异常
错误作为 tool message 回传给模型
```

### 模型无限调用工具

当前解法：

```text
MAX_AGENT_STEPS 限制最大 Agent loop 轮数
```

### RAG 检索不到资料

当前解法：

```text
RAG 工具返回 found=false 和 result_count=0
system prompt 要求没有资料时如实说明
```

### RAG 结果来源不清楚

当前解法：

```text
每个 chunk 都带 source 和 chunk_id
模型回答知识库问题时可以提及来源
```

### embedding 重复计算浪费成本

当前解法：

```text
embedding_cache.json 缓存文本 embedding
vector_index.json 缓存 chunk + embedding + metadata
```

### Agent 行为缺少评估

当前解法：

```text
eval_cases.json 固化典型问题和预期结果
scripts/run_eval.py 检查 Tool Router、关键词 RAG、结构化数据工具和 memory 裁剪
第一版 eval 不依赖 LLM，保证便宜、稳定、可重复
```

### Agent 只能在命令行使用

当前解法：

```text
app.py 提供 FastAPI HTTP 接口
GET /health 用于健康检查
POST /chat 用于外部系统或前端调用 Agent
session_store.py 用内存字典按 session_id 保存 messages，支持多轮 Web 聊天
```

### 多轮聊天导致 messages 无限增长

当前解法：

```text
memory.py 在调用模型前后裁剪 messages
保留 system message 和最近几轮完整对话
MAX_CONTEXT_MESSAGES 控制最大 message 数量
```

## 适合继续学习的方向

下一步可以继续做：

- 接入真正的向量数据库
- 增加 LLM 工具选择 eval
- 把 fake_db 替换成真实数据库或外部 API

现在项目已经从单文件教学版，升级成了按职责拆分的应用版结构。
