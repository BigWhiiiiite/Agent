const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chat-form");
const inputEl = document.querySelector("#message-input");
const sendButtonEl = document.querySelector("#send-button");
const sessionInputEl = document.querySelector("#session-id");
const roleSelectEl = document.querySelector("#user-role");
const pageSelectEl = document.querySelector("#page-context");
const resetButtonEl = document.querySelector("#reset-session");
const statusPillEl = document.querySelector("#status-pill");
const quickPromptEls = document.querySelectorAll("[data-prompt]");
const traceContentEl = document.querySelector("#trace-content");
const clearTraceEl = document.querySelector("#clear-trace");

let isSending = false;

function setStatus(text, state = "ready") {
  statusPillEl.classList.remove("busy", "error");

  if (state !== "ready") {
    statusPillEl.classList.add(state);
  }

  statusPillEl.lastChild.textContent = ` ${text}`;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function createMessage(role, content = "") {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "U" : "A";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  if (role === "user") {
    article.append(bubble, avatar);
  } else {
    article.append(avatar, bubble);
  }

  messagesEl.append(article);
  scrollToBottom();
  return bubble;
}

function createErrorMessage(content) {
  const article = document.createElement("article");
  article.className = "message assistant error";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "!";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  article.append(avatar, bubble);
  messagesEl.append(article);
  scrollToBottom();
}

function createTextElement(tagName, className, text) {
  const element = document.createElement(tagName);
  element.className = className;
  element.textContent = text;
  return element;
}

function createJsonBlock(value) {
  const block = document.createElement("pre");
  block.className = "trace-json";
  block.textContent = JSON.stringify(value, null, 2);
  return block;
}

function renderTrace(trace) {
  traceContentEl.innerHTML = "";

  if (!trace) {
    traceContentEl.append(createTextElement("p", "trace-empty", "No trace"));
    return;
  }

  const selectedToolsGroup = document.createElement("section");
  selectedToolsGroup.className = "trace-group";
  selectedToolsGroup.append(createTextElement("div", "trace-label", "Selected tools"));

  const chips = document.createElement("div");
  chips.className = "trace-chips";
  const selectedTools = trace.selected_tools && trace.selected_tools.length
    ? trace.selected_tools
    : ["none"];

  selectedTools.forEach((toolName) => {
    chips.append(createTextElement("span", "trace-chip", toolName));
  });

  selectedToolsGroup.append(chips);
  traceContentEl.append(selectedToolsGroup);

  const metaGroup = document.createElement("section");
  metaGroup.className = "trace-group";
  metaGroup.append(createTextElement("div", "trace-label", "Stop reason"));
  metaGroup.append(createTextElement("span", "trace-chip", trace.stop_reason || "unknown"));
  traceContentEl.append(metaGroup);

  const toolCallsGroup = document.createElement("section");
  toolCallsGroup.className = "trace-group";
  toolCallsGroup.append(createTextElement("div", "trace-label", "Tool calls"));

  if (!trace.tool_calls || trace.tool_calls.length === 0) {
    toolCallsGroup.append(createTextElement("p", "trace-empty", "No tool call"));
  } else {
    trace.tool_calls.forEach((toolCall, index) => {
      const step = document.createElement("article");
      step.className = "trace-step";

      const title = document.createElement("div");
      title.className = "trace-step-title";
      title.append(createTextElement("span", "", `${index + 1}. ${toolCall.tool_name}`));

      const ok = toolCall.result_summary && toolCall.result_summary.ok;
      title.append(createTextElement("span", "trace-step-status", ok ? "ok" : "error"));

      step.append(title);
      step.append(createTextElement("div", "trace-label", "Arguments"));
      step.append(createJsonBlock(toolCall.arguments || {}));
      step.append(createTextElement("div", "trace-label", "Result"));
      step.append(createJsonBlock(toolCall.result_summary || {}));
      toolCallsGroup.append(step);
    });
  }

  traceContentEl.append(toolCallsGroup);
}

function buildPayload(message, reset = false) {
  return {
    session_id: sessionInputEl.value.trim() || "default",
    message,
    reset,
    context: {
      user_role: roleSelectEl.value,
      page: pageSelectEl.value
    }
  };
}

function parseSseBlock(block) {
  const eventLine = block
    .split("\n")
    .find((line) => line.startsWith("event:"));
  const dataLine = block
    .split("\n")
    .find((line) => line.startsWith("data:"));

  if (!eventLine || !dataLine) {
    return null;
  }

  return {
    event: eventLine.replace("event:", "").trim(),
    data: JSON.parse(dataLine.replace("data:", "").trim())
  };
}

async function readStream(response, assistantBubble) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let doneData = null;

  while (true) {
    const { value, done } = await reader.read();

    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop();

    for (const block of blocks) {
      const parsed = parseSseBlock(block);

      if (!parsed) {
        continue;
      }

      if (parsed.event === "delta") {
        assistantBubble.textContent += parsed.data.content;
        scrollToBottom();
      }

      if (parsed.event === "error") {
        throw new Error(parsed.data.message);
      }

      if (parsed.event === "done") {
        doneData = parsed.data;
      }
    }
  }

  return doneData;
}

async function sendMessage(message) {
  if (isSending || !message.trim()) {
    return;
  }

  isSending = true;
  sendButtonEl.disabled = true;
  resetButtonEl.disabled = true;
  setStatus("Thinking", "busy");

  createMessage("user", message);
  const assistantBubble = createMessage("assistant", "");

  try {
    const response = await fetch("/chat/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(buildPayload(message))
    });

    if (!response.ok || !response.body) {
      throw new Error(`HTTP ${response.status}`);
    }

    const doneData = await readStream(response, assistantBubble);

    if (!assistantBubble.textContent.trim()) {
      assistantBubble.textContent = "没有返回内容。";
    }

    renderTrace(doneData ? doneData.trace : null);
    setStatus("Ready");
  } catch (error) {
    assistantBubble.closest(".message").remove();
    createErrorMessage(error.message);
    setStatus("Error", "error");
  } finally {
    isSending = false;
    sendButtonEl.disabled = false;
    resetButtonEl.disabled = false;
    inputEl.focus();
  }
}

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  inputEl.value = "";
  sendMessage(message);
});

quickPromptEls.forEach((button) => {
  button.addEventListener("click", () => {
    inputEl.value = button.dataset.prompt;
    inputEl.focus();
  });
});

resetButtonEl.addEventListener("click", async () => {
  if (isSending) {
    return;
  }

  const sessionId = sessionInputEl.value.trim() || "default";
  setStatus("Resetting", "busy");

  try {
    await fetch(`/sessions/${encodeURIComponent(sessionId)}`, {
      method: "DELETE"
    });

    messagesEl.innerHTML = "";
    renderTrace(null);
    createMessage(
      "assistant",
      "你好，我可以查询老师时间、课程信息，也可以回答知识库里的制度和 Agent 学习问题。"
    );
    setStatus("Ready");
  } catch (error) {
    createErrorMessage(error.message);
    setStatus("Error", "error");
  }
});

clearTraceEl.addEventListener("click", () => {
  renderTrace(null);
});
