async function postForm(url, formData) {
  try {
    const res = await fetch(url, { method: 'POST', body: formData });
    return await res.json();
  } catch (e) {
    return { error: e.message || String(e) };
  }
}

async function postJSON(url, obj) {
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(obj)
    });
    return await res.json();
  } catch (e) {
    return { error: e.message || String(e) };
  }
}

const uploadForm = document.getElementById('uploadForm');
const status = document.getElementById('status');
const sessionUI = document.getElementById('sessionUI');
const sessionIdSpan = document.getElementById('sessionId');
const chatDiv = document.getElementById('chat');
const spinner = document.getElementById('spinner');

let session_id = null;

uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  const filesEl = document.getElementById('files');
  if (filesEl.files.length === 0) {
    alert('Choose at least one PDF');
    return;
  }

  const fd = new FormData();
  for (let f of filesEl.files) fd.append('files', f);

  // disable UI while uploading
  uploadForm.querySelector('button[type=submit]').disabled = true;
  filesEl.disabled = true;
  spinner.style.display = 'inline-block';
  status.innerText = 'Uploading and processing...';

  const res = await postForm('/upload', fd);

  // re-enable UI
  uploadForm.querySelector('button[type=submit]').disabled = false;
  filesEl.disabled = false;
  spinner.style.display = 'none';

  if (res.error) {
    status.innerText = 'Error: ' + res.error;
    return;
  }

  session_id = res.session_id || 'default';
  sessionIdSpan.innerText = session_id;
  sessionUI.style.display = 'block';
  status.innerText = 'PDF processed. You can chat now.';
});

document.getElementById('send').addEventListener('click', async () => {
  const q = document.getElementById('question').value;
  if (!q || !session_id) return;

  appendChat('You', q, 'user');
  document.getElementById('question').value = '';

  const thinking = appendChat('Bot', 'Thinking...', 'bot');

  const res = await postJSON('/chat', { message: q });

  thinking.remove();

  if (res.error) {
    appendChat('Error', res.error, 'bot');
  } else {
    appendChat('Bot', res.response, 'bot');
  }
});

document.getElementById('abort').addEventListener('click', async () => {
  if (!confirm('This will delete all stored vectors. Continue?')) return;

  const abortBtn = document.getElementById('abort');
  abortBtn.disabled = true;
  spinner.style.display = 'inline-block';

  const res = await postJSON('/abort', {});

  // re-enable UI
  abortBtn.disabled = false;
  spinner.style.display = 'none';

  // server returns { messages: [...] }
  if (res && Array.isArray(res.messages)) {
    appendChat('System', 'Session aborted. Data deleted.', 'bot');
    session_id = null;
    // reload so the UI is fully reset and fresh
    location.reload();
  } else {
    appendChat('Error', res.error || 'Unknown error', 'bot');
  }
});

function appendChat(who, text, cls) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.innerText = who + ': ' + text;
  chatDiv.appendChild(d);
  chatDiv.scrollTop = chatDiv.scrollHeight;
  return d;
}