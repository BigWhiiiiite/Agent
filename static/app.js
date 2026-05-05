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
    }
  }
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

    await readStream(response, assistantBubble);

    if (!assistantBubble.textContent.trim()) {
      assistantBubble.textContent = "没有返回内容。";
    }

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
