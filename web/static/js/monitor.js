/** monitor.js — Live Monitor page logic */

updateClock("mon-time");

let alertCount      = 0;
let currentSource   = "camera";
let uploadedVideoPath = null;

// ── WebSocket callbacks ───────────────────────────────────────────────────────
window.onFpsUpdate = function (data) {
  document.getElementById("mon-fps").textContent   = data.fps;
  document.getElementById("mon-faces").textContent = data.faces;
  const wEl = document.getElementById("mon-weapons");
  wEl.textContent   = data.weapons;
  wEl.style.color   = data.weapons > 0 ? "var(--red)" : "var(--text-primary)";
  document.getElementById("mon-source").textContent = (data.source || "--")
    .replace("camera:", "Cam ")
    .replace("file:", "📁 ");

  const from_config = document.getElementById("mon-engine");
  // Engine displayed from initial status:init or show from api
};

window.onFaceAlert = function (data) {
  addAlert(buildFaceAlertCard(data));
  // Flash threat banner for criminal detection
  flashThreat(`⚠ CRIMINAL DETECTED: ${data.name}`);
};

window.onWeaponAlert = function (data) {
  addAlert(buildWeaponAlertCard(data));
  if (data.threat_level === "CRITICAL" || data.threat_level === "HIGH") {
    flashThreat(`🔫 ${data.threat_level} THREAT: ${data.weapon_types}`);
  }
};

window.onFeedStarted = function () {
  document.getElementById("mon-status-badge").style.display = "inline-flex";
  document.getElementById("btn-start").disabled  = true;
  document.getElementById("btn-stop").disabled   = false;
  document.getElementById("btn-pause").disabled  = false;
  const nBtn = document.getElementById("btn-native");
  nBtn.style.display = "inline-flex";
  nBtn.disabled = false;
  nBtn.classList.replace("btn-outline", "btn-primary");
  nBtn.textContent = "💻 Switch to Web Feed";
};

window.onFeedStopped = function () {
  document.getElementById("mon-status-badge").style.display = "none";
  document.getElementById("btn-start").disabled  = false;
  document.getElementById("btn-stop").disabled   = true;
  document.getElementById("btn-pause").disabled  = true;
  document.getElementById("btn-resume").style.display = "none";
  document.getElementById("btn-pause").style.display  = "inline-flex";
  
  const nBtn = document.getElementById("btn-native");
  nBtn.style.display = "none";
  nBtn.disabled = true;
  if(nBtn.classList.contains("btn-primary")) {
     nBtn.classList.replace("btn-primary", "btn-outline");
     nBtn.textContent = "🖥️ Native Feed";
  }
};

socket.on("status:init", (data) => {
  document.getElementById("mon-engine").textContent = data.engine || "--";
  setSidebarStatus(true, data.model_loaded);
  if (data.feed_status === "running") {
    window.onFeedStarted();
  }
});

// ── Feed controls ─────────────────────────────────────────────────────────────
async function startMonitorFeed() {
  let body;
  if (currentSource === "camera") {
    const idx = parseInt(document.getElementById("cam-index").value) || 0;
    body = { source: "camera", camera_index: idx };
  } else {
    if (!uploadedVideoPath) {
      showToast("Please upload a video file first", "warning");
      return;
    }
    body = { source: "video", path: uploadedVideoPath };
  }
  const res = await apiPost("/api/feed/start", body);
  if (!res.success) showToast(res.error || "Failed to start feed", "error");
  else showToast("📹 Feed started", "success");
}

async function stopMonitorFeed() {
  const res = await apiPost("/api/feed/stop");
  if (res.success) showToast("Feed stopped", "info");
}

async function pauseFeed() {
  await apiPost("/api/feed/pause");
  document.getElementById("btn-pause").style.display  = "none";
  document.getElementById("btn-resume").style.display = "inline-flex";
}

async function resumeFeed() {
  await apiPost("/api/feed/resume");
  document.getElementById("btn-resume").style.display = "none";
  document.getElementById("btn-pause").style.display  = "inline-flex";
}

function toggleNativeFeed() {
  socket.emit("feed:control", { action: "toggle_native" });
}

socket.on("status:native_window", (data) => {
  const btn = document.getElementById("btn-native");
  if (data.active) {
    btn.classList.replace("btn-outline", "btn-primary");
    btn.textContent = "💻 Switch to Web Feed";
  } else {
    btn.classList.replace("btn-primary", "btn-outline");
    btn.textContent = "🖥️ Switch to Native Feed";
  }
});

// -- Focus Mode & Anti-Spoofing ------------------------------------------------
async function setFocusMode(mode) {
  const res = await apiPost("/api/mode", { mode });
  if (res.success) {
    showToast(`Mode switched to: ${res.data.label}`, "success");
    // Update the antispoof toggle enabled state based on if weapon mode is active
    document.getElementById("antispoof-toggle").disabled = !res.data.weapon_enabled;
  } else {
    showToast(res.error || "Failed to switch mode", "error");
  }
}

async function toggleAntiSpoof(enabled) {
  const res = await apiPost("/api/antispoof/toggle", { enabled });
  if (res.success) {
    showToast(res.data.message, "success");
  } else {
    showToast(res.error || "Failed to toggle anti-spoofing", "error");
    document.getElementById("antispoof-toggle").checked = !enabled; // Revert UI
  }
}

socket.on("mode:switching", (data) => {
  showToast("Switching detection mode...", "info");
  document.getElementById("focus-mode-select").disabled = true;
});

socket.on("mode:changed", (data) => {
  document.getElementById("focus-mode-select").value = data.mode;
  document.getElementById("focus-mode-select").disabled = false;
  document.getElementById("antispoof-toggle").disabled = !data.weapon_enabled;
});

// -- Source switch -------------------------------------------------------------
function selectSource(src) {
  currentSource = src;
  document.getElementById("src-camera").classList.toggle("active", src === "camera");
  document.getElementById("src-video").classList.toggle("active",  src === "video");
  document.getElementById("cam-options").style.display   = src === "camera" ? "block" : "none";
  document.getElementById("video-options").style.display = src === "video"  ? "block" : "none";
}

// ── Drag-and-drop video upload ────────────────────────────────────────────────
const dropZone = document.getElementById("drop-zone");

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) uploadVideoFile(file);
});

function handleVideoFileSelect(event) {
  const file = event.target.files[0];
  if (file) uploadVideoFile(file);
}

async function uploadVideoFile(file) {
  const allowedExts = [".mp4", ".avi", ".mov", ".mkv"];
  const ext = "." + file.name.split(".").pop().toLowerCase();
  if (!allowedExts.includes(ext)) {
    showToast(`Unsupported format: ${ext}`, "error");
    return;
  }

  document.getElementById("upload-progress").style.display = "block";
  document.getElementById("drop-zone-text").textContent    = `Uploading: ${file.name}`;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/upload/video", { method: "POST", body: formData });
    const data = await res.json();

    document.getElementById("upload-progress").style.display = "none";

    if (data.success) {
      uploadedVideoPath = data.data.path;
      document.getElementById("drop-zone-text").innerHTML =
        `✅ <strong>${file.name}</strong><br/>${data.data.size_mb} MB — Ready`;
      document.getElementById("up-prog-bar").style.width = "100%";
      showToast(`Video uploaded: ${file.name}`, "success");
    } else {
      showToast(data.error || "Upload failed", "error");
      document.getElementById("drop-zone-text").innerHTML =
        "Drop video file here<br/>or click to select<br/><span class='text-muted'>MP4, AVI, MOV, MKV</span>";
    }
  } catch (err) {
    document.getElementById("upload-progress").style.display = "none";
    showToast("Upload error: " + err.message, "error");
  }
}

// ── Alerts panel ──────────────────────────────────────────────────────────────
function addAlert(html) {
  alertCount++;
  document.getElementById("alert-count").textContent = alertCount;

  const panel = document.getElementById("monitor-alerts");
  const empty  = document.getElementById("alert-empty");
  if (empty) empty.remove();

  panel.insertAdjacentHTML("afterbegin", html);

  // Cap at 30 alerts in panel
  const cards = panel.querySelectorAll(".alert-card");
  if (cards.length > 30) cards[cards.length - 1].remove();
}

function clearAlerts() {
  const panel = document.getElementById("monitor-alerts");
  panel.innerHTML = `<div class="empty-state" id="alert-empty" style="padding:var(--sp-xl)">
    <div class="empty-state-icon">🔕</div><div class="empty-state-text">No alerts yet</div></div>`;
  alertCount = 0;
  document.getElementById("alert-count").textContent = "0";
}

// ── Threat banner ─────────────────────────────────────────────────────────────
let threatTimer = null;
function flashThreat(message) {
  const banner = document.getElementById("threat-banner");
  const text   = document.getElementById("threat-text");
  text.textContent = message;
  banner.classList.add("show");
  if (threatTimer) clearTimeout(threatTimer);
  threatTimer = setTimeout(() => banner.classList.remove("show"), 5000);
}

// ── Check initial feed status ─────────────────────────────────────────────────
(async () => {
  try {
    const res = await apiGet("/api/feed/status");
    if (res.data?.status === "running") window.onFeedStarted();
  } catch (_) {}
})();
