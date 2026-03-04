document.addEventListener("DOMContentLoaded", () => {
  console.log("app.js loaded"); // Debugging Log, feel free to delete


  // Variables for DOM
  const chat = document.getElementById("chat");
  const form = document.getElementById("chatForm");
  const input = document.getElementById("message");

  const landing = document.getElementById("landing");
  const chatScreen = document.getElementById("chatScreen");
  const startBtn = document.getElementById("startBtn");
  const backToLanding = document.getElementById("backToLanding");
  const clearChat = document.getElementById("clearChat");


  // Theme toggle button and root element for theme switching
  const themeToggle = document.getElementById("themeToggle");
  const root = document.documentElement;

  // --- Theme (Just Dark/Light Mode, fairly simple) ---
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
  function showChat() {
    landing.classList.add("hidden");
    chatScreen.classList.remove("hidden");
    input.focus();
  }

  function showLanding() {
    chatScreen.classList.add("hidden");
    landing.classList.remove("hidden");
  }

  startBtn.addEventListener("click", showChat);
  backToLanding.addEventListener("click", showLanding);

  // --- Chat helpers ---
  function addBubble(text, who) {
    const div = document.createElement("div");
    div.className = `bubble ${who}`;
    div.textContent = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
  }

  function clearChatUI() {
    chat.innerHTML = "";
  }

  clearChat.addEventListener("click", clearChatUI);


  // --- Form submit ---
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = input.value.trim();
    if (!message) return;

    addBubble(message, "user");
    input.value = "";

    // lightweight client-side guardrail examples (this is a stub)
    const lower = message.toLowerCase();
    if (lower.includes("what should we do") || lower.includes("tell me what to do")) {
      addBubble(
        "I can’t make decisions or prescribe actions. If you share your location and time period, I can summarise the forecast implications and key considerations.",
        "bot"
      );
      return;
    }

    // loading indicator
    const loadingId = `loading-${Date.now()}`;
    const loadingDiv = document.createElement("div");
    loadingDiv.className = "bubble bot";
    loadingDiv.id = loadingId;
    loadingDiv.textContent = "Thinking…";
    chat.appendChild(loadingDiv);
    chat.scrollTop = chat.scrollHeight;

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ message })
      });

      const data = await res.json();

      // remove loading
      const toRemove = document.getElementById(loadingId);
      if (toRemove) toRemove.remove();

      if (!res.ok) {
        addBubble(data.reply || "Something went wrong.", "bot");
        return;
      }
      addBubble(data.reply, "bot");
    } catch {
      const toRemove = document.getElementById(loadingId);
      if (toRemove) toRemove.remove();
      addBubble("Network error. Try again.", "bot");
    }
  });
});
