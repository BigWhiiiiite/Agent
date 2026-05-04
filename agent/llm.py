from openai import OpenAI

from agent.config import MODEL_NAME


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
