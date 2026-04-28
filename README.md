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

## 文件结构

```text
.
├── main.py
├── requirements.txt
└── docs/
    ├── agent_notes.txt
    ├── course_rules.txt
    └── school_rules.txt
```

`main.py` 是主程序。

`docs/` 是本地知识库，RAG 工具会从这里检索资料。

程序运行过程中可能生成：

```text
embedding_cache.json
vector_index.json
```

这两个文件是本地缓存，已经放进 `.gitignore`，不会提交到 GitHub。

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
```

## 运行

```bash
python3 main.py
```

输入 `exit` 退出。

## 当前工具

### query_teacher_schedule

查询某位老师在指定日期的空闲时间。

示例问题：

```text
帮我看看李老师周五什么时候有空
```

### query_course_info

查询课程介绍、授课老师、时间和教室。

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
while True:
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

## 适合继续学习的方向

下一步可以继续做：

- 接入真正的向量数据库
- 增加工具错误处理
- 增加测试用例
- 把工具和 schema 拆成多个文件

现在这个项目保持在单文件结构，是为了让初学者能完整看懂 Agent 骨架。
