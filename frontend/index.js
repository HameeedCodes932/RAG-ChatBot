/**
 * RAG Chatbot — Frontend Logic
 * Handles file upload, chat interaction, and UI state management.
 */

// ── State ────────────────────────────────────────────────────────────
const state = {
  sessionId: crypto.randomUUID(),
  isProcessing: false,
  documentsUploaded: 0,
};

// ── DOM Elements ─────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const els = {
  uploadZone: $('upload-zone'),
  fileInput: $('file-input'),
  uploadProgress: $('upload-progress'),
  progressFill: $('progress-fill'),
  progressText: $('progress-text'),
  documentsList: $('documents-list'),
  emptyDocs: $('empty-docs'),
  statusDot: $('status-dot'),
  statusText: $('status-text'),
  messagesContainer: $('messages-container'),
  welcomeScreen: $('welcome-screen'),
  typingIndicator: $('typing-indicator'),
  chatInput: $('chat-input'),
  btnSend: $('btn-send'),
  btnClearHistory: $('btn-clear-history'),
  chatSubtitle: $('chat-subtitle'),
  toastContainer: $('toast-container'),
  mobileToggle: $('mobile-toggle'),
  sidebar: $('sidebar'),
};

// ── API Helpers ──────────────────────────────────────────────────────
const API_BASE = '';

async function apiPost(endpoint, body, isFormData = false) {
  const options = {
    method: 'POST',
    body: isFormData ? body : JSON.stringify(body),
  };
  if (!isFormData) {
    options.headers = { 'Content-Type': 'application/json' };
  }
  const res = await fetch(`${API_BASE}${endpoint}`, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function apiGet(endpoint) {
  const res = await fetch(`${API_BASE}${endpoint}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ── Toast Notifications ──────────────────────────────────────────────
function showToast(message, type = 'info', duration = 4000) {
  const icons = { success: '✅', error: '❌', info: 'ℹ️' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type] || ''}</span><span>${message}</span>`;
  els.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = 'toastOut 0.3s ease forwards';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ── File Upload ──────────────────────────────────────────────────────
function initUpload() {
  // Click to browse
  els.uploadZone.addEventListener('click', () => els.fileInput.click());

  // File selected
  els.fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) uploadFile(e.target.files[0]);
  });

  // Drag and drop
  els.uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    els.uploadZone.classList.add('drag-over');
  });

  els.uploadZone.addEventListener('dragleave', () => {
    els.uploadZone.classList.remove('drag-over');
  });

  els.uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    els.uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
  });
}

async function uploadFile(file) {
  const fileNameLower = file.name.toLowerCase();
  const allowedExtensions = ['.txt', '.pdf', '.docx'];
  const isValid = allowedExtensions.some(ext => fileNameLower.endsWith(ext));

  if (!isValid) {
    showToast('Only .txt, .pdf, and .docx files are supported', 'error');
    return;
  }

  // Show progress
  els.uploadProgress.classList.add('active');
  els.progressFill.style.width = '30%';
  els.progressText.textContent = 'Uploading...';

  try {
    const formData = new FormData();
    formData.append('file', file);

    els.progressFill.style.width = '60%';
    els.progressText.textContent = 'Processing chunks...';

    const result = await apiPost('/api/upload', formData, true);

    els.progressFill.style.width = '100%';
    els.progressText.textContent = 'Done!';

    // Add document to sidebar list
    addDocumentToList(result);
    state.documentsUploaded++;
    updateChatSubtitle();

    showToast(`"${result.filename}" ingested — ${result.chunks} chunks`, 'success');

    // Hide welcome screen after first upload
    if (els.welcomeScreen) {
      els.welcomeScreen.style.display = 'none';
    }
  } catch (err) {
    showToast(`Upload failed: ${err.message}`, 'error');
  } finally {
    setTimeout(() => {
      els.uploadProgress.classList.remove('active');
      els.progressFill.style.width = '0%';
    }, 1500);
    els.fileInput.value = '';
  }
}

function addDocumentToList(doc) {
  els.emptyDocs.style.display = 'none';

  const item = document.createElement('div');
  item.className = 'doc-item';
  item.innerHTML = `
    <div class="doc-icon">📄</div>
    <div class="doc-info">
      <div class="doc-name" title="${doc.filename}">${doc.filename}</div>
      <div class="doc-meta">${doc.chunks} chunks · ${formatBytes(doc.characters)}</div>
    </div>
  `;
  els.documentsList.appendChild(item);
}

function formatBytes(chars) {
  if (chars < 1024) return `${chars} chars`;
  return `${(chars / 1024).toFixed(1)}K chars`;
}

// ── Chat ─────────────────────────────────────────────────────────────
function initChat() {
  // Send on button click
  els.btnSend.addEventListener('click', sendMessage);

  // Send on Enter (Shift+Enter for newline)
  els.chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Auto-resize textarea
  els.chatInput.addEventListener('input', () => {
    els.chatInput.style.height = 'auto';
    els.chatInput.style.height = Math.min(els.chatInput.scrollHeight, 140) + 'px';
  });

  // Clear history
  els.btnClearHistory.addEventListener('click', clearChat);
}

async function sendMessage() {
  const question = els.chatInput.value.trim();
  if (!question || state.isProcessing) return;

  state.isProcessing = true;
  els.btnSend.disabled = true;

  // Hide welcome screen
  if (els.welcomeScreen) {
    els.welcomeScreen.style.display = 'none';
  }

  // Add user message
  appendMessage('user', question);

  // Clear input
  els.chatInput.value = '';
  els.chatInput.style.height = 'auto';

  // Show typing indicator
  els.typingIndicator.classList.add('active');
  scrollToBottom();

  try {
    const result = await apiPost('/api/chat', {
      question: question,
      session_id: state.sessionId,
    });

    // Update session ID
    state.sessionId = result.session_id;

    // Hide typing indicator and show response
    els.typingIndicator.classList.remove('active');
    appendMessage('bot', result.answer, result.sources);
  } catch (err) {
    els.typingIndicator.classList.remove('active');
    appendMessage('bot', `⚠️ Error: ${err.message}. Make sure LM Studio is running.`);
    showToast('Failed to get response', 'error');
  } finally {
    state.isProcessing = false;
    els.btnSend.disabled = false;
    els.chatInput.focus();
  }
}

function appendMessage(role, content, sources = []) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role}`;

  const avatar = role === 'user' ? '👤' : '🤖';

  let sourcesHtml = '';
  if (sources && sources.length > 0) {
    const chips = sources
      .map((s) => `<span class="source-chip">📎 ${s.source}</span>`)
      .join('');
    sourcesHtml = `<div class="message-sources">${chips}</div>`;
  }

  // Sanitize content (basic XSS prevention) and preserve newlines
  const sanitized = content
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>');

  msgDiv.innerHTML = `
    <div class="message-avatar">${avatar}</div>
    <div class="message-content">
      ${sanitized}
      ${sourcesHtml}
    </div>
  `;

  // Insert before typing indicator
  els.messagesContainer.insertBefore(msgDiv, els.typingIndicator);
  scrollToBottom();
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    els.messagesContainer.scrollTop = els.messagesContainer.scrollHeight;
  });
}

async function clearChat() {
  // Clear messages from DOM
  const messages = els.messagesContainer.querySelectorAll('.message');
  messages.forEach((msg) => msg.remove());

  // Show welcome screen if no messages
  if (els.welcomeScreen) {
    els.welcomeScreen.style.display = 'flex';
  }

  // Clear server-side history
  try {
    await apiPost(`/api/clear-history?session_id=${state.sessionId}`, {});
  } catch {
    // Ignore errors on clear
  }

  // New session
  state.sessionId = crypto.randomUUID();
  showToast('Chat history cleared', 'info');
}

// ── Health Check ─────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const health = await apiGet('/api/health');
    const llm = health.llm;

    if (llm.status === 'connected') {
      els.statusDot.className = 'status-dot connected';
      els.statusText.textContent = `LM Studio · Connected`;
    } else {
      els.statusDot.className = 'status-dot disconnected';
      els.statusText.textContent = 'LM Studio · Disconnected';
      showToast('LM Studio is not reachable. Start the server on port 1234.', 'error', 6000);
    }

    // Load existing documents
    if (health.documents_ingested > 0) {
      loadDocuments();
    }
  } catch {
    els.statusDot.className = 'status-dot disconnected';
    els.statusText.textContent = 'Backend · Offline';
  }
}

async function loadDocuments() {
  try {
    const data = await apiGet('/api/documents');
    if (data.documents.length > 0) {
      els.emptyDocs.style.display = 'none';
      data.documents.forEach(addDocumentToList);
      state.documentsUploaded = data.documents.length;
      updateChatSubtitle();
    }
  } catch {
    // Silently fail
  }
}

function updateChatSubtitle() {
  els.chatSubtitle.textContent =
    state.documentsUploaded > 0
      ? `${state.documentsUploaded} document${state.documentsUploaded > 1 ? 's' : ''} loaded · Ready to chat`
      : 'Upload a document to start';
}

// ── Mobile Toggle ────────────────────────────────────────────────────
function initMobile() {
  els.mobileToggle.addEventListener('click', () => {
    els.sidebar.classList.toggle('open');
  });

  // Close sidebar on outside click (mobile)
  document.addEventListener('click', (e) => {
    if (
      window.innerWidth <= 768 &&
      els.sidebar.classList.contains('open') &&
      !els.sidebar.contains(e.target) &&
      e.target !== els.mobileToggle
    ) {
      els.sidebar.classList.remove('open');
    }
  });
}

// ── Initialize ───────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initUpload();
  initChat();
  initMobile();
  checkHealth();
});
