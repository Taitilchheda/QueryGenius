const state = {
  token: localStorage.getItem("qg_token") || "",
  user: null,
  currentChatId: null,
  messages: [],
  showArchived: false,
};

const prefs = {
  topK: Number.parseInt(localStorage.getItem("qg_pref_topk") || "5", 10),
  mode: localStorage.getItem("qg_pref_mode") || "balanced",
};

const els = {
  fileInput: document.getElementById("fileInput"),
  uploadBtn: document.getElementById("uploadBtn"),
  ingestBtn: document.getElementById("ingestBtn"),
  refreshDocsBtn: document.getElementById("refreshDocsBtn"),
  uploadStatus: document.getElementById("uploadStatus"),
  docsList: document.getElementById("docsList"),

  newSessionBtn: document.getElementById("newSessionBtn"),
  toggleArchiveBtn: document.getElementById("toggleArchiveBtn"),
  sessionsList: document.getElementById("sessionsList"),
  sessionLabel: document.getElementById("sessionLabel"),

  chatMessages: document.getElementById("chatMessages"),
  chatForm: document.getElementById("chatForm"),
  questionInput: document.getElementById("questionInput"),
  topKInput: document.getElementById("topKInput"),
  profileInput: document.getElementById("profileInput"),

  accountBtn: document.getElementById("accountBtn"),
  accountMenu: document.getElementById("accountMenu"),
  accountUserPane: document.getElementById("accountUserPane"),
  accountGuestPane: document.getElementById("accountGuestPane"),
  menuSettingsBtn: document.getElementById("menuSettingsBtn"),
  menuLoginBtn: document.getElementById("menuLoginBtn"),
  menuRegisterBtn: document.getElementById("menuRegisterBtn"),
  userLabel: document.getElementById("userLabel"),
  logoutBtn: document.getElementById("logoutBtn"),

  authModal: document.getElementById("authModal"),
  authModalClose: document.getElementById("authModalClose"),
  authModalTitle: document.getElementById("authModalTitle"),
  emailInput: document.getElementById("emailInput"),
  passwordInput: document.getElementById("passwordInput"),
  loginBtn: document.getElementById("loginBtn"),
  registerBtn: document.getElementById("registerBtn"),
  authStatus: document.getElementById("authStatus"),

  settingsModal: document.getElementById("settingsModal"),
  settingsModalClose: document.getElementById("settingsModalClose"),
  settingsTopK: document.getElementById("settingsTopK"),
  settingsMode: document.getElementById("settingsMode"),
  saveSettingsBtn: document.getElementById("saveSettingsBtn"),

  diagramZoomModal: document.getElementById("diagramZoomModal"),
  diagramZoomBody: document.getElementById("diagramZoomBody"),
  diagramZoomClose: document.getElementById("diagramZoomClose"),
};

function authHeaders(extra = {}) {
  const headers = { ...extra };
  if (state.token) {
    headers.Authorization = `Bearer ${state.token}`;
  }
  return headers;
}

function shortId(id) {
  return id ? id.slice(0, 8) : "none";
}

function setStatus(text, ok = true) {
  els.uploadStatus.textContent = text;
  els.uploadStatus.className = ok ? "status ok" : "status error";
}

function setAuthStatus(text, ok = true) {
  els.authStatus.textContent = text;
  els.authStatus.className = ok ? "status ok" : "status error";
}

function updateSessionLabel() {
  els.sessionLabel.textContent = shortId(state.currentChatId);
}

function openAuthModal(mode = "signin") {
  els.authModalTitle.textContent = mode === "register" ? "Register" : "Sign In";
  els.authModal.classList.remove("hidden");
  els.emailInput.focus();
}

function closeAuthModal() {
  els.authModal.classList.add("hidden");
  els.authStatus.textContent = "";
}

function openSettingsModal() {
  els.settingsTopK.value = String(Number.isFinite(prefs.topK) ? prefs.topK : 5);
  els.settingsMode.value = prefs.mode || "balanced";
  els.settingsModal.classList.remove("hidden");
}

function closeSettingsModal() {
  els.settingsModal.classList.add("hidden");
}

function openDiagramZoom(node) {
  if (!node || !els.diagramZoomBody || !els.diagramZoomModal) return;
  els.diagramZoomBody.innerHTML = "";
  const clone = node.cloneNode(true);
  els.diagramZoomBody.appendChild(clone);
  els.diagramZoomModal.classList.remove("hidden");
}

function closeDiagramZoom() {
  els.diagramZoomModal.classList.add("hidden");
  els.diagramZoomBody.innerHTML = "";
}

function toggleAccountMenu() {
  els.accountMenu.classList.toggle("hidden");
}

function closeAccountMenu() {
  els.accountMenu.classList.add("hidden");
}

function refreshAccountView() {
  const authed = !!state.token && !!state.user;
  if (authed) {
    els.accountBtn.textContent = `Account (${state.user.email.split("@")[0]})`;
    els.accountUserPane.classList.remove("hidden");
    els.accountGuestPane.classList.add("hidden");
    els.userLabel.textContent = state.user.email;
  } else {
    els.accountBtn.textContent = "Account";
    els.accountUserPane.classList.add("hidden");
    els.accountGuestPane.classList.remove("hidden");
    els.userLabel.textContent = "";
  }
}

function applyPrefs() {
  const topK = Number.isFinite(prefs.topK) ? Math.min(20, Math.max(1, prefs.topK)) : 5;
  const mode = prefs.mode || "balanced";
  els.topKInput.value = String(topK);
  els.profileInput.value = mode;
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data);
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return data;
}

function normalizeEmail(value) {
  return (value || "").trim().toLowerCase();
}

function validateAuthInputs() {
  const email = normalizeEmail(els.emailInput.value);
  const password = els.passwordInput.value || "";
  if (!email || !password) {
    return { ok: false, message: "Email and password are required." };
  }
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return { ok: false, message: "Enter a valid email address." };
  }
  if (password.length < 6) {
    return { ok: false, message: "Password must be at least 6 characters." };
  }
  return { ok: true, email, password };
}

function mapAuthError(errorMessage, action) {
  const msg = String(errorMessage || "").toLowerCase();
  if (msg.includes("422") || msg.includes("unprocessable")) {
    return "Invalid input. Use a valid email and password (6+ chars).";
  }
  if (msg.includes("invalid email or password") || msg.includes("401")) {
    return action === "login"
      ? "Login failed. Check email/password."
      : "Account exists, but password did not match.";
  }
  if (msg.includes("already registered") || msg.includes("409")) {
    return "Email is already registered. Please sign in.";
  }
  return errorMessage;
}

function escapeHtml(str) {
  return str
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatPlainMarkdown(text) {
  const placeholders = [];
  let staged = text;

  const preserve = (pattern, block = false) => {
    staged = staged.replace(pattern, (m) => {
      const token = `@@MATH_${placeholders.length}@@`;
      const safe = escapeHtml(m);
      placeholders.push({
        token,
        html: block ? `<div class="math-block">${safe}</div>` : `<span class="math-inline">${safe}</span>`,
      });
      return token;
    });
  };

  // Preserve math segments before markdown/newline transforms.
  preserve(/\\\[[\s\S]*?\\\]/g, true); // \[ ... \]
  preserve(/\$\$[\s\S]*?\$\$/g, true); // $$ ... $$
  preserve(/\\\([\s\S]*?\\\)/g, false); // \( ... \)
  preserve(/\$(?!\$)([^$\n]|\\\$)+\$/g, false); // $ ... $

  let html = escapeHtml(staged);
  html = html.replace(/^### (.*)$/gm, "<h4>$1</h4>");
  html = html.replace(/^## (.*)$/gm, "<h3>$1</h3>");
  html = html.replace(/^\d+\.\s+(.*)$/gm, "• $1");
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\n/g, "<br>");

  for (const item of placeholders) {
    html = html.replaceAll(item.token, item.html);
  }
  return html;
}

function renderAssistantHtml(text) {
  const source = text || "";
  const codeRegex = /```([a-zA-Z0-9_-]+)?\r?\n([\s\S]*?)```/g;
  const parts = [];
  let cursor = 0;
  let match;
  while ((match = codeRegex.exec(source)) !== null) {
    const before = source.slice(cursor, match.index);
    if (before.trim()) {
      parts.push(`<div class="prose">${formatPlainMarkdown(before)}</div>`);
    }
    const lang = (match[1] || "").toLowerCase();
    const code = match[2] || "";
    if (lang === "mermaid") {
      parts.push(`<pre class="code-block code-mermaid"><code>mermaid\n${escapeHtml(code.trim())}</code></pre>`);
    } else {
      parts.push(`<pre class="code-block"><code>${escapeHtml(code.trim())}</code></pre>`);
    }
    cursor = match.index + match[0].length;
  }
  const tail = source.slice(cursor);
  if (tail.trim()) {
    parts.push(`<div class="prose">${formatPlainMarkdown(tail)}</div>`);
  }
  return parts.join("");
}

function renderMath(container) {
  // First, render explicit placeholder nodes for guaranteed LaTeX display.
  const mathNodes = container.querySelectorAll(".math-block, .math-inline");
  if (window.katex && mathNodes.length) {
    for (const node of mathNodes) {
      const raw = (node.textContent || "").trim();
      if (!raw) continue;
      let expr = raw;
      let displayMode = node.classList.contains("math-block");
      if (raw.startsWith("\\[") && raw.endsWith("\\]")) {
        expr = raw.slice(2, -2).trim();
        displayMode = true;
      } else if (raw.startsWith("$$") && raw.endsWith("$$")) {
        expr = raw.slice(2, -2).trim();
        displayMode = true;
      } else if (raw.startsWith("\\(") && raw.endsWith("\\)")) {
        expr = raw.slice(2, -2).trim();
        displayMode = false;
      } else if (raw.startsWith("$") && raw.endsWith("$") && raw.length > 2) {
        expr = raw.slice(1, -1).trim();
        displayMode = false;
      }
      if (!expr) continue;
      try {
        node.innerHTML = window.katex.renderToString(expr, {
          throwOnError: false,
          displayMode,
          strict: "ignore",
        });
      } catch (_) {}
    }
  }

  // Then run auto-render for any remaining delimiters.
  if (window.renderMathInElement) {
    try {
      window.renderMathInElement(container, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true },
        ],
        throwOnError: false,
      });
    } catch (_) {}
  }
}

function extractFlowLabels(raw) {
  const nodeMap = new Map();
  const nodeRegex = /([A-Za-z0-9_]+)\s*\[\s*\"?([^\]\"]+)\"?\s*\]/g;
  let nm;
  while ((nm = nodeRegex.exec(raw)) !== null) {
    nodeMap.set(nm[1], nm[2].trim());
  }
  const edgeRegex = /([A-Za-z0-9_]+)\s*-->\s*([A-Za-z0-9_]+)/g;
  const edges = [];
  const indeg = new Map();
  let em;
  while ((em = edgeRegex.exec(raw)) !== null) {
    const a = em[1];
    const b = em[2];
    edges.push([a, b]);
    indeg.set(b, (indeg.get(b) || 0) + 1);
    if (!indeg.has(a)) indeg.set(a, 0);
  }
  if (!edges.length) return Array.from(nodeMap.values()).slice(0, 10);
  let start = edges[0][0];
  for (const [node, d] of indeg.entries()) {
    if (d === 0) {
      start = node;
      break;
    }
  }
  const adj = new Map();
  edges.forEach(([a, b]) => {
    if (!adj.has(a)) adj.set(a, []);
    adj.get(a).push(b);
  });
  const labels = [];
  const seen = new Set();
  let cur = start;
  while (cur && !seen.has(cur) && labels.length < 12) {
    seen.add(cur);
    labels.push(nodeMap.get(cur) || cur);
    cur = (adj.get(cur) || [])[0];
  }
  return labels;
}

function renderMermaidFallback(container) {
  const blocks = container.querySelectorAll("pre.code-mermaid code, .mermaid-wrap");
  for (const item of blocks) {
    let raw = "";
    if (item.matches("pre.code-mermaid code")) {
      raw = (item.textContent || "").replace(/^mermaid\s*\n?/i, "").trim();
    } else {
      raw = item.dataset.rawMermaid || "";
    }
    const labels = extractFlowLabels(raw);
    if (!labels.length) continue;
    const wrap = document.createElement("div");
    wrap.className = "diagram-fallback";
    labels.forEach((label, idx) => {
      const node = document.createElement("div");
      node.className = "diagram-node";
      node.textContent = label;
      wrap.appendChild(node);
      if (idx < labels.length - 1) {
        const arrow = document.createElement("div");
        arrow.className = "diagram-arrow";
        arrow.textContent = "→";
        wrap.appendChild(arrow);
      }
    });
    const target = item.matches("pre.code-mermaid code") ? item.closest("pre") : item;
    if (target) target.replaceWith(wrap);
  }
}

async function upgradeMermaidBlocks(container) {
  const mermaidBlocks = container.querySelectorAll("pre.code-mermaid code");
  if (!mermaidBlocks.length) return;
  if (!window.mermaid) {
    renderMermaidFallback(container);
    return;
  }
  try {
    window.mermaid.initialize({ startOnLoad: false, theme: "default", securityLevel: "loose" });
  } catch (_) {
    renderMermaidFallback(container);
    return;
  }
  for (const codeEl of mermaidBlocks) {
    const full = codeEl.textContent || "";
    const raw = full.replace(/^mermaid\s*\n?/i, "").trim();
    if (!raw) continue;
    const wrap = document.createElement("div");
    wrap.className = "mermaid-wrap";
    wrap.dataset.rawMermaid = raw;
    const mer = document.createElement("div");
    mer.className = "mermaid";
    mer.textContent = raw;
    wrap.appendChild(mer);
    const pre = codeEl.closest("pre");
    if (pre) pre.replaceWith(wrap);
  }
  try {
    if (typeof window.mermaid.run === "function") {
      await window.mermaid.run({ querySelector: ".mermaid" });
    } else if (typeof window.mermaid.init === "function") {
      window.mermaid.init(undefined, container.querySelectorAll(".mermaid"));
    }
  } catch (_) {
    renderMermaidFallback(container);
  }
}

function addMessage(role, html, meta = "", citations = "") {
  const item = document.createElement("article");
  item.className = `msg ${role}`;
  item.innerHTML = `<div class="meta">${meta}</div><div>${html}</div>${citations ? `<div class="citations">${citations}</div>` : ""}`;
  els.chatMessages.appendChild(item);
  if (role === "assistant") {
    upgradeMermaidBlocks(item);
    renderMath(item);
  }
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
  return item;
}

function renderMessages() {
  els.chatMessages.innerHTML = "";
  for (const message of state.messages) {
    if (message.role === "user") {
      addMessage("user", escapeHtml(message.question), `You • ${message.created_at}`);
      continue;
    }
    const citationText = (message.citations || []).map((c) => `${c.source}:${c.chunk_id} (score ${c.score})`).join("\n");
    addMessage(
      "assistant",
      renderAssistantHtml(message.answer),
      `Assistant • ${message.created_at} • ${message.latency_ms} ms • mode ${message.retrieval_profile || "balanced"}`,
      citationText
    );
  }
}

async function typeAssistantMessage(text, meta, citations = "") {
  const item = document.createElement("article");
  item.className = "msg assistant";
  item.innerHTML = `<div class="meta">${escapeHtml(meta)}</div><div class="typed"></div>${citations ? `<div class="citations">${escapeHtml(citations)}</div>` : ""}`;
  els.chatMessages.appendChild(item);
  const typedEl = item.querySelector(".typed");
  const safeText = escapeHtml(text);
  let idx = 0;
  while (idx < safeText.length) {
    typedEl.innerHTML = safeText.slice(0, idx);
    idx += 4;
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
    await new Promise((resolve) => setTimeout(resolve, 8));
  }
  typedEl.innerHTML = renderAssistantHtml(text);
  await upgradeMermaidBlocks(item);
  renderMath(item);
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

async function loadDocuments() {
  const data = await fetchJSON("/documents");
  els.docsList.innerHTML = "";
  if (!data.documents.length) {
    els.docsList.innerHTML = "<li>No documents found.</li>";
    return;
  }
  for (const doc of data.documents) {
    const li = document.createElement("li");
    li.textContent = `${doc.name} • ${Math.round(doc.size_bytes / 1024)} KB`;
    els.docsList.appendChild(li);
  }
}

async function loadChats() {
  if (!state.token) {
    els.sessionsList.innerHTML = "<li>Sign in to save chats. You can still ask as guest.</li>";
    return;
  }
  const data = await fetchJSON(`/chats?archived=${state.showArchived}`, { headers: authHeaders() });
  const chats = data.chats || [];
  els.sessionsList.innerHTML = "";
  if (!chats.length) {
    els.sessionsList.innerHTML = "<li>No chats yet.</li>";
    return;
  }

  for (const chat of chats) {
    const updated = new Date(chat.last_updated || chat.created_at || Date.now());
    const updatedText = `${updated.toLocaleDateString()} ${updated.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
    const li = document.createElement("li");
    if (chat.session_id === state.currentChatId) li.classList.add("active");
    li.innerHTML = `
      <div class="session-row">
        <div class="session-title">${escapeHtml(chat.title || shortId(chat.session_id))}</div>
        <span class="session-actions">
          <button class="icon-btn chat-archive" title="Archive">🗄</button>
          <button class="icon-btn chat-delete" title="Delete">🗑</button>
        </span>
      </div>
      <div class="session-meta">${chat.turns} turns • ${escapeHtml(updatedText)}</div>
    `;

    li.onclick = async (event) => {
      const target = event.target;
      if (target.classList.contains("chat-archive") || target.classList.contains("chat-delete")) {
        return;
      }
      await openChat(chat.session_id);
    };

    li.querySelector(".chat-archive").onclick = async (event) => {
      event.stopPropagation();
      await fetchJSON(`/chats/${chat.session_id}`, {
        method: "PATCH",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ archived: !chat.archived }),
      });
      if (chat.session_id === state.currentChatId && !chat.archived) {
        state.currentChatId = null;
        state.messages = [];
        renderMessages();
        updateSessionLabel();
      }
      await loadChats();
    };

    li.querySelector(".chat-delete").onclick = async (event) => {
      event.stopPropagation();
      await fetchJSON(`/chats/${chat.session_id}`, {
        method: "DELETE",
        headers: authHeaders(),
      });
      if (chat.session_id === state.currentChatId) {
        state.currentChatId = null;
        state.messages = [];
        renderMessages();
        updateSessionLabel();
      }
      await loadChats();
    };

    els.sessionsList.appendChild(li);
  }
}

async function openChat(chatId) {
  const data = await fetchJSON(`/chats/${chatId}/messages`, { headers: authHeaders() });
  state.currentChatId = chatId;
  state.messages = [];
  for (const row of data.messages) {
    state.messages.push({ role: "user", question: row.question, created_at: row.created_at });
    state.messages.push({
      role: "assistant",
      answer: row.answer,
      citations: row.citations || [],
      latency_ms: row.latency_ms,
      retrieval_profile: row.retrieval_profile || "balanced",
      created_at: row.created_at,
    });
  }
  updateSessionLabel();
  renderMessages();
  await loadChats();
}

async function createNewChat() {
  if (!state.token) {
    state.currentChatId = crypto.randomUUID();
    state.messages = [];
    updateSessionLabel();
    renderMessages();
    return;
  }
  const data = await fetchJSON("/chats", {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ title: "New chat" }),
  });
  state.currentChatId = data.session_id;
  state.messages = [];
  updateSessionLabel();
  renderMessages();
  await loadChats();
}

async function uploadFiles() {
  const files = els.fileInput.files;
  if (!files.length) {
    setStatus("Pick one or more files first.", false);
    return;
  }
  const form = new FormData();
  for (const file of files) form.append("files", file);
  setStatus("Uploading files...");
  const data = await fetchJSON("/upload", { method: "POST", body: form });
  setStatus(`Uploaded: ${data.uploaded.length}, skipped: ${data.skipped.length}`, true);
  els.fileInput.value = "";
  await loadDocuments();
}

async function rebuildIndex() {
  setStatus("Building index...");
  const data = await fetchJSON("/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rebuild: true, chunk_size_chars: 1500, overlap_chars: 200 }),
  });
  setStatus(`Indexed ${data.num_chunks} chunks.`, true);
}

async function sendQuestion(event) {
  event.preventDefault();
  const question = els.questionInput.value.trim();
  if (!question) return;

  const topK = Math.max(1, Math.min(20, Number.parseInt(els.topKInput.value || "5", 10)));
  const profile = (els.profileInput.value || "balanced").trim();

  if (!state.currentChatId) {
    await createNewChat();
  }

  const createdAt = new Date().toISOString();
  state.messages.push({ role: "user", question, created_at: createdAt });
  addMessage("user", escapeHtml(question), `You • ${createdAt}`);
  els.questionInput.value = "";
  autoResizeComposer();

  const thinkingNode = addMessage("assistant", "Thinking...", "Assistant");

  try {
    const response = await fetchJSON("/chat", {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        question,
        top_k: topK,
        session_id: state.currentChatId,
        retrieval_profile: profile,
      }),
    });

    thinkingNode.remove();
    state.currentChatId = response.session_id;
    updateSessionLabel();

    const assistantMsg = {
      role: "assistant",
      answer: response.answer,
      citations: response.citations || [],
      latency_ms: response.latency_ms,
      retrieval_profile: response.retrieval_profile || profile,
      created_at: response.created_at,
    };
    state.messages.push(assistantMsg);

    const citationText = (assistantMsg.citations || []).map((c) => `${c.source}:${c.chunk_id} (score ${c.score})`).join("\n");
    await typeAssistantMessage(
      assistantMsg.answer,
      `Assistant • ${assistantMsg.created_at} • ${assistantMsg.latency_ms} ms • mode ${assistantMsg.retrieval_profile}`,
      citationText
    );

    await loadChats();
  } catch (err) {
    thinkingNode.remove();
    await typeAssistantMessage(`Error: ${err.message}`, `Assistant • ${new Date().toISOString()}`);
  }
}

function autoResizeComposer() {
  const el = els.questionInput;
  el.style.height = "auto";
  const max = 180;
  el.style.height = `${Math.min(el.scrollHeight, max)}px`;
}

async function onLogin() {
  const validated = validateAuthInputs();
  if (!validated.ok) {
    setAuthStatus(validated.message, false);
    return;
  }

  let data;
  try {
    data = await fetchJSON("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: validated.email, password: validated.password }),
    });
  } catch (err) {
    setAuthStatus(mapAuthError(err.message, "login"), false);
    return;
  }

  state.token = data.token;
  state.user = data.user;
  localStorage.setItem("qg_token", state.token);
  els.passwordInput.value = "";
  setAuthStatus("Logged in.", true);
  refreshAccountView();
  closeAuthModal();
  closeAccountMenu();
  await loadChats();
}

async function onRegister() {
  const validated = validateAuthInputs();
  if (!validated.ok) {
    setAuthStatus(validated.message, false);
    return;
  }

  try {
    const data = await fetchJSON("/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: validated.email, password: validated.password }),
    });
    state.token = data.token;
    state.user = data.user;
    localStorage.setItem("qg_token", state.token);
    els.passwordInput.value = "";
    setAuthStatus("Registered and logged in.", true);
    refreshAccountView();
    closeAuthModal();
    closeAccountMenu();
    await loadChats();
  } catch (err) {
    setAuthStatus(mapAuthError(err.message, "register"), false);
  }
}

async function logout() {
  try {
    await fetchJSON("/auth/logout", { method: "POST", headers: authHeaders() });
  } catch (_) {}
  state.token = "";
  state.user = null;
  state.currentChatId = null;
  state.messages = [];
  localStorage.removeItem("qg_token");
  refreshAccountView();
  closeAccountMenu();
  renderMessages();
  updateSessionLabel();
  await loadChats();
}

async function loadMe() {
  if (!state.token) {
    state.user = null;
    refreshAccountView();
    return;
  }
  try {
    const user = await fetchJSON("/auth/me", { headers: authHeaders() });
    state.user = user;
  } catch (_) {
    state.token = "";
    state.user = null;
    localStorage.removeItem("qg_token");
  }
  refreshAccountView();
}

function saveSettings() {
  const topK = Math.max(1, Math.min(20, Number.parseInt(els.settingsTopK.value || "5", 10)));
  const mode = (els.settingsMode.value || "balanced").trim();
  prefs.topK = topK;
  prefs.mode = mode;
  localStorage.setItem("qg_pref_topk", String(topK));
  localStorage.setItem("qg_pref_mode", mode);
  applyPrefs();
  closeSettingsModal();
}

function bindEvents() {
  els.uploadBtn.onclick = () => uploadFiles().catch((err) => setStatus(err.message, false));
  els.ingestBtn.onclick = () => rebuildIndex().catch((err) => setStatus(err.message, false));
  els.refreshDocsBtn.onclick = () => loadDocuments().catch((err) => setStatus(err.message, false));

  els.newSessionBtn.onclick = () => createNewChat().catch((err) => setStatus(err.message, false));
  els.toggleArchiveBtn.onclick = () => {
    state.showArchived = !state.showArchived;
    els.toggleArchiveBtn.textContent = state.showArchived ? "Active" : "Archived";
    loadChats().catch((err) => setStatus(err.message, false));
  };

  els.chatForm.addEventListener("submit", sendQuestion);
  els.questionInput.addEventListener("input", autoResizeComposer);
  els.questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      els.chatForm.requestSubmit();
    }
  });

  els.accountBtn.onclick = (event) => {
    event.stopPropagation();
    toggleAccountMenu();
  };

  els.menuLoginBtn.onclick = () => {
    closeAccountMenu();
    openAuthModal("signin");
  };

  els.menuRegisterBtn.onclick = () => {
    closeAccountMenu();
    openAuthModal("register");
  };

  els.menuSettingsBtn.onclick = () => {
    closeAccountMenu();
    openSettingsModal();
  };

  els.logoutBtn.onclick = () => logout().catch((err) => setAuthStatus(err.message, false));

  els.authModalClose.onclick = closeAuthModal;
  els.settingsModalClose.onclick = closeSettingsModal;
  els.diagramZoomClose.onclick = closeDiagramZoom;
  els.loginBtn.onclick = () => onLogin().catch((err) => setAuthStatus(err.message, false));
  els.registerBtn.onclick = () => onRegister().catch((err) => setAuthStatus(err.message, false));
  els.saveSettingsBtn.onclick = saveSettings;

  els.chatMessages.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const diagramNode = target.closest(".mermaid-wrap, .diagram-fallback");
    if (diagramNode) {
      openDiagramZoom(diagramNode);
    }
  });

  document.addEventListener("click", (event) => {
    const target = event.target;
    if (!els.accountMenu.classList.contains("hidden") && !els.accountMenu.contains(target) && !els.accountBtn.contains(target)) {
      closeAccountMenu();
    }
    if (!els.authModal.classList.contains("hidden") && target === els.authModal) {
      closeAuthModal();
    }
    if (!els.settingsModal.classList.contains("hidden") && target === els.settingsModal) {
      closeSettingsModal();
    }
    if (!els.diagramZoomModal.classList.contains("hidden") && target === els.diagramZoomModal) {
      closeDiagramZoom();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      if (!els.diagramZoomModal.classList.contains("hidden")) closeDiagramZoom();
      if (!els.settingsModal.classList.contains("hidden")) closeSettingsModal();
      if (!els.authModal.classList.contains("hidden")) closeAuthModal();
      if (!els.accountMenu.classList.contains("hidden")) closeAccountMenu();
    }
  });
}

async function init() {
  bindEvents();
  applyPrefs();
  autoResizeComposer();
  await loadMe();
  await loadDocuments();
  await loadChats();
  updateSessionLabel();

  // If KaTeX arrives late from CDN, re-render existing assistant math content once.
  window.addEventListener("load", () => {
    const assistantMessages = document.querySelectorAll(".msg.assistant");
    for (const msg of assistantMessages) {
      renderMath(msg);
    }
  });
}

init().catch((err) => setStatus(err.message, false));
