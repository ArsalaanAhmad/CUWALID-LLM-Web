document.addEventListener("DOMContentLoaded", () => {
  console.log("app.js loaded");

  // Core DOM
  const chat = document.getElementById("chat");
  const form = document.getElementById("chatForm");
  const input = document.getElementById("message");

  const landing = document.getElementById("landing");
  const chatScreen = document.getElementById("chatScreen");
  const startBtn = document.getElementById("startBtn");
  const backToLanding = document.getElementById("backToLanding");
  const clearChat = document.getElementById("clearChat");
  const chatTabs = document.getElementById("chatTabs");
  const addChatBtn = document.getElementById("addChatBtn");

  // Audience + mode
  const audienceChips = document.querySelectorAll(".audience-chip");
  const modeChips = document.querySelectorAll(".mode-chip");
  const examplesList = document.getElementById("examplesList");

  // Theme
  const themeToggle = document.getElementById("themeToggle");
  const root = document.documentElement;

  // App state
  let selectedAudience = "general-public";
  let selectedMode = "forecast";
  const CHAT_SESSIONS_KEY = "chat_sessions";
  const ACTIVE_CHAT_ID_KEY = "active_chat_id";
  const CHAT_TRANSCRIPTS_KEY = "chat_transcripts";
  const MAX_CHATS = 3;
  let chats = [];
  let activeChatId = "";
  let chatTranscripts = {};
  // Optional bot-only typewriter effect (disabled for users who prefer reduced motion)
  const enableTypewriter = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const examplePrompts = {
    forecast: [
      "I am in Garissa, Kenya. Send me the flood forecast in Swahili for OND 2026.",
      "Give me the crop forecast for Marsabit, Kenya for OND 2026 in English.",
      "What is the pasture outlook for Somalia in MAM 2026?"
    ],
    modelling: [
      "Show me the CUWALID workflow from climate forcing to hydrological outputs.",
      "What documentation sections should I read first to deploy CUWALID in a new basin?",
      "What model inputs and preprocessing steps are required before running DRYP in CUWALID?"
    ]
  };

  // --- Theme ---
  const savedTheme = localStorage.getItem("theme");
  if (savedTheme) root.setAttribute("data-theme", savedTheme);

  function syncThemeLabel() {
    const current = root.getAttribute("data-theme") || "dark";
    themeToggle.textContent = current === "dark" ? "Light" : "Dark";
  }

  syncThemeLabel();

  themeToggle.addEventListener("click", () => {
    const current = root.getAttribute("data-theme") || "dark";
    const next = current === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    syncThemeLabel();
  });

  // --- View toggling ---
  function createConversationId() {
    return crypto.randomUUID();
  }

  function saveChatState() {
    localStorage.setItem(CHAT_SESSIONS_KEY, JSON.stringify(chats));
    localStorage.setItem(ACTIVE_CHAT_ID_KEY, activeChatId);
    localStorage.setItem(CHAT_TRANSCRIPTS_KEY, JSON.stringify(chatTranscripts));
  }

  function saveActiveTranscript() {
    ensureActiveChat();
    chatTranscripts[activeChatId] = chat.innerHTML;
  }

  function restoreActiveTranscript() {
    ensureActiveChat();
    chat.innerHTML = chatTranscripts[activeChatId] || "";
    chat.scrollTop = chat.scrollHeight;
  }

  function createChatName(index) {
    return `Chat ${index}`;
  }

  function ensureActiveChat() {
    const activeExists = chats.some((c) => c.id === activeChatId);
    if (activeExists) return;

    if (chats.length > 0) {
      activeChatId = chats[0].id;
      return;
    }

    const first = { id: createConversationId(), name: "Chat 1" };
    chats = [first];
    activeChatId = first.id;
  }

  function renderChatTabs() {
    if (!chatTabs) return;

    chatTabs.innerHTML = "";

    chats.forEach((chatSession) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chat-tab";
      if (chatSession.id === activeChatId) {
        btn.classList.add("active");
      }

      btn.textContent = chatSession.name;
      btn.addEventListener("click", () => {
        if (chatSession.id === activeChatId) return;

        saveActiveTranscript();
        activeChatId = chatSession.id;
        saveChatState();
        restoreActiveTranscript();
        renderChatTabs();
      });

      chatTabs.appendChild(btn);
    });
  }

  function loadChatState() {
    const storedChatsRaw = localStorage.getItem(CHAT_SESSIONS_KEY);
    const storedActiveId = localStorage.getItem(ACTIVE_CHAT_ID_KEY);
    const storedTranscriptsRaw = localStorage.getItem(CHAT_TRANSCRIPTS_KEY);

    if (storedChatsRaw) {
      try {
        const parsed = JSON.parse(storedChatsRaw);
        if (Array.isArray(parsed)) {
          chats = parsed
            .filter((item) => item && typeof item.id === "string" && typeof item.name === "string")
            .slice(0, MAX_CHATS);
        }
      } catch (err) {
        console.warn("Failed to parse chat sessions from localStorage", err);
      }
    }

    if (storedTranscriptsRaw) {
      try {
        const parsed = JSON.parse(storedTranscriptsRaw);
        if (parsed && typeof parsed === "object") {
          chatTranscripts = parsed;
        }
      } catch (err) {
        console.warn("Failed to parse chat transcripts from localStorage", err);
      }
    }

    activeChatId = storedActiveId || "";
    ensureActiveChat();
    if (!chatTranscripts[activeChatId]) {
      chatTranscripts[activeChatId] = "";
    }
    saveChatState();
    renderChatTabs();
    restoreActiveTranscript();
  }

  function addChatSession() {
    if (chats.length >= MAX_CHATS) {
      alert("Maximum 3 chats allowed");
      return;
    }

    const newChat = {
      id: createConversationId(),
      name: createChatName(chats.length + 1)
    };

    chats.push(newChat);
    activeChatId = newChat.id;
    chatTranscripts[activeChatId] = "";
    saveChatState();
    restoreActiveTranscript();
    renderChatTabs();
  }

  function resetActiveChatSession() {
    ensureActiveChat();
    const idx = chats.findIndex((c) => c.id === activeChatId);
    if (idx === -1) {
      return;
    }

    chats[idx] = {
      ...chats[idx],
      id: createConversationId(),
    };
    activeChatId = chats[idx].id;
    chatTranscripts[activeChatId] = "";
    saveChatState();
    restoreActiveTranscript();
    renderChatTabs();
  }

  function showChat() {
    landing.classList.add("hidden");
    chatScreen.classList.remove("hidden");
    input.focus();
  }

  function showLanding() {
    chatScreen.classList.add("hidden");
    landing.classList.remove("hidden");
  }

  startBtn.addEventListener("click", () => {
    // Starting from landing explicitly resets the active chat context.
    resetActiveChatSession();
    showChat();
  });
  backToLanding.addEventListener("click", showLanding);
  if (addChatBtn) {
    addChatBtn.addEventListener("click", addChatSession);
  }

  // --- Example prompts ---
  function renderExamplePrompts() {
    if (!examplesList) return;

    examplesList.innerHTML = "";

    const prompts = examplePrompts[selectedMode] || [];
    prompts.forEach((promptText) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "example-chip";
      btn.textContent = promptText;

      btn.addEventListener("click", () => {
        input.value = promptText;
        input.focus();
      });

      examplesList.appendChild(btn);
    });

    input.placeholder =
      selectedMode === "forecast"
        ? "Ask about a forecast..."
        : "Ask about CUWALID setup, workflows, or documentation...";
  }

  // --- Chat helpers ---
  function addBubble(text, who) {
    const div = document.createElement("div");
    div.className = `bubble ${who}`;
    div.textContent = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    saveActiveTranscript();
  }

  function addLoadingBubble(id) {
    const div = document.createElement("div");
    div.className = "bubble bot";
    div.id = id;

    div.innerHTML = `
    <div class="loading-bubble">
      <span class="loading-spinner" aria-hidden="true"></span>
      <span class="loading-text">Thinking…</span>
    </div>
  `;

    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    saveActiveTranscript();
  }

  function typewriterBubble(text, who = "bot", speed = 12) {
    return new Promise((resolve) => {
      const div = document.createElement("div");
      div.className = `bubble ${who}`;
      div.textContent = "";
      chat.appendChild(div);

      let i = 0;

      function step() {
        if (i < text.length) {
          div.textContent += text.charAt(i);
          i += 1;
          chat.scrollTop = chat.scrollHeight;
          saveActiveTranscript();
          setTimeout(step, speed);
        } else {
          saveActiveTranscript();
          resolve();
        }
      }

      step();
    });
  }

  async function renderBotReply(replyText) {
    const safeText = replyText || "No response received.";

    if (enableTypewriter) {
      await typewriterBubble(safeText, "bot", 10);
      return;
    }

    addBubble(safeText, "bot");
  }

  function addAttachment(url, label = "Open attachment") {
    const wrap = document.createElement("div");
    wrap.className = "attachment-row";

    const link = document.createElement("a");
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.className = "attachment-link";
    link.textContent = label;

    wrap.appendChild(link);
    chat.appendChild(wrap);
    chat.scrollTop = chat.scrollHeight;
    saveActiveTranscript();
  }

  function renderAttachments(attachments) {
    if (!Array.isArray(attachments)) return;

    attachments.forEach((item) => {
      if (!item?.url) return;

      const label =
        item.label ||
        (item.type === "map"
          ? "Open forecast map"
          : item.type === "source"
          ? "Open source"
          : "Open attachment");

      addAttachment(item.url, label);
    });
  }

  function isObviousClientSideRefusal(message) {
    if (selectedMode !== "forecast") return false;

    const lower = message.toLowerCase();
    return lower.includes("what should we do") || lower.includes("tell me what to do");
  }

  function buildChatPayload(message) {
    ensureActiveChat();
    return {
      message,
      mode: selectedMode,
      tier: selectedAudience,
      conversation_id: activeChatId
    };
  }

  function removeLoadingBubble(id) {
    const toRemove = document.getElementById(id);
    if (toRemove) toRemove.remove();
  }

  function clearChatUI() {
    chat.innerHTML = "";
    saveActiveTranscript();
  }

  clearChat.addEventListener("click", resetActiveChatSession);

  // --- Audience chips ---
  audienceChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      audienceChips.forEach((c) => c.classList.remove("is-active"));
      chip.classList.add("is-active");
      selectedAudience = chip.dataset.audience;
    });
  });

  // --- Mode chips ---
  modeChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      modeChips.forEach((c) => c.classList.remove("is-active"));
      chip.classList.add("is-active");
      selectedMode = chip.dataset.mode;
      renderExamplePrompts();
    });
  });

  // --- Form submit ---
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const message = input.value.trim();
    if (!message) return;

    addBubble(message, "user");
    input.value = "";

    // Frontend performs only lightweight refusal for very obvious phrases.
    // All substantive guardrails are enforced by the backend.
    if (isObviousClientSideRefusal(message)) {
      addBubble(
        "I can’t make decisions or prescribe actions. If you share your location and time period, I can summarise the forecast implications and key considerations.",
        "bot"
      );
      return;
    }

    // Loading bubble
    const loadingId = `loading-${Date.now()}`;
    addLoadingBubble(loadingId);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildChatPayload(message))
      });

      const data = await res.json();

      removeLoadingBubble(loadingId);

      if (!res.ok) {
        addBubble(data.reply || "Something went wrong.", "bot");
        return;
      }

      await renderBotReply(data.reply);
      renderAttachments(data.attachments);
    } catch (err) {
      removeLoadingBubble(loadingId);

      addBubble("Network error. Try again.", "bot");
      console.error(err);
    }
  });

  loadChatState();
  renderExamplePrompts();
});