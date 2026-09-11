import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "CRT_MiniMaxH3UnifiedSampler";
const NODE_ALIASES = new Set(["MiniMax H3 Unified Sampler (CRT)", "CRT_MiniMaxH3UnifiedSampler"]);
const CONFIG_NODE_NAME = "CRT_MiniMaxH3USConfig";
const STYLE_ID = "crt-h3us-unified-sampler-v3";
const MIN_WIDTH = 450;
const MIN_HEIGHT = 1;

// --- KJNodes-style animated preview override transport --------------------
// The backend encodes one animated WebP per sampling step and sends it as
// base64 JSON; the browser loops it natively in an <img>, which reads as a
// real video instead of the frontend's replace-per-message still previews.

function h3usFindNodeByQualifiedId(rootGraph, qid) {
  if (!rootGraph || !qid) return null;
  const parts = String(qid).split(":");
  let graph = rootGraph;
  for (let i = 0; i < parts.length - 1; i++) {
    const parentId = parseInt(parts[i], 10);
    if (!Number.isFinite(parentId)) return null;
    const parentNode = graph?.getNodeById?.(parentId);
    if (!parentNode?.subgraph) return null;
    graph = parentNode.subgraph;
  }
  const leafId = parseInt(parts[parts.length - 1], 10);
  if (!Number.isFinite(leafId)) return null;
  return graph?.getNodeById?.(leafId) || null;
}

function h3usB64ToBlob(b64, mime) {
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return new Blob([arr], { type: mime });
}

function h3usBuildPreviewWidget(node) {
  const wrap = document.createElement("div");
  wrap.style.cssText =
    "position:relative;width:100%;min-height:140px;background:#070707;" +
    "border-radius:6px;overflow:hidden;border:1px solid #1c1c1c;";
  const img = document.createElement("img");
  img.style.cssText =
    "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:none;";
  img.draggable = false;
  wrap.appendChild(img);
  const placeholder = document.createElement("div");
  placeholder.textContent = "live preview idle";
  placeholder.style.cssText =
    "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;" +
    "color:#5a5a5a;font-size:12px;";
  wrap.appendChild(placeholder);

  node.addDOMWidget("h3us_live_preview", "preview", wrap, { serialize: false });

  let currentUrl = null;

  const handler = (data) => {
    try {
      if (!data) return;
      if (Array.isArray(data.sigmas)) {
        // New run: drop the previous run's animation immediately.
        if (currentUrl) {
          URL.revokeObjectURL(currentUrl);
          currentUrl = null;
        }
        img.style.display = "none";
        img.removeAttribute("src");
        placeholder.style.display = "flex";
      }
      if (typeof data.image !== "string") return;
      const blob = h3usB64ToBlob(data.image, data.mime || "image/webp");
      const url = URL.createObjectURL(blob);
      if (Number.isFinite(data.w) && Number.isFinite(data.h) && data.w > 0 && data.h > 0) {
        wrap.style.aspectRatio = `${data.w} / ${data.h}`;
      }
      const promote = () => {
        if (currentUrl === url && img.src !== url) {
          img.src = url;
        } else {
          img.src = url;
        }
        img.style.display = "block";
        placeholder.style.display = "none";
        const previous = currentUrl;
        currentUrl = url;
        if (previous && previous !== url) {
          window.setTimeout(() => URL.revokeObjectURL(previous), 500);
        }
      };
      if (typeof img.decode === "function") {
        img.addEventListener("load", promote, { once: true });
        img.src = url;
      } else {
        img.src = url;
        promote();
      }
      // Mirror into the PREVIEW tab image when that view exists.
      const tabImg = node.h3usUI?.previewImage;
      if (tabImg && tabImg !== img) {
        tabImg.src = url;
        tabImg.style.display = "block";
      }
    } catch (err) {
      console.warn("[CRT H3] preview override decode failed:", err);
    }
  };

  node._crtH3PreviewHandler = handler;
  node._h3usPovCleanup = () => {
    if (currentUrl) URL.revokeObjectURL(currentUrl);
    currentUrl = null;
    node._crtH3PreviewHandler = null;
  };
}

api.addEventListener("crt_minimaxh3_preview", (e) => {
  const data = e.detail;
  if (!data || data.node_id == null) return;
  const node = h3usFindNodeByQualifiedId(app.graph, data.node_id);
  if (node?._crtH3PreviewHandler) node._crtH3PreviewHandler(data);
});


const WORKFLOW_MODES = ["T2V", "I2V", "R2V"];

const MODE_FIELDS = {
  "T2V": ["aspect_ratio"],
  "I2V": ["fl_aspect_mode"],
  "R2V": ["aspect_ratio"],
};

const SETTINGS_SECTIONS = [
  {
    id: "INFERENCE",
    title: "Inference",
    fields: ["steps", "steps_turbo", "shift_video", "shift_audio", "megapixels_target", "length_frames", "audio_frames_override", "video_frames_override"],
  },
  {
    id: "SPEED",
    title: "Speed Optimizations",
    fields: ["turbo", "enable_sol_attn", "enable_chunk_ff", "enable_spectrum"],
  },
  {
    id: "OUTPUT",
    title: "Output",
    fields: ["vae_decode_tiled", "unload_before_decode", "low_vram", "generated_audio_gain_db"],
  },
  {
    id: "DEROPE",
    title: "Motion Smearing Fix",
    fields: ["enable_derope", "enable_derope_adapter", "derope_preset", "derope_audio", "derope_mode", "derope_profile", "derope_inject", "derope_abstain", "derope_protect_tail"],
  },
  {
    id: "NEGPIP",
    title: "NegPiP",
    fields: ["enable_negpip"],
  },
  {
    id: "PREVIEW",
    title: "Preview",
    kind: "preview-row",
  },
];

const FIELD_LABELS = {
  aspect_ratio: "Aspect",
  fl_aspect_mode: "If F/L aspect differs",
  steps: "Steps",
  steps_turbo: "Steps Turbo",
  shift_video: "Sigma Shift (video)",
  shift_audio: "Sigma Shift (audio)",
  turbo: "Turbo",
  enable_sol_attn: "Sol Attention",
  enable_chunk_ff: "Chunk FeedForward",
  enable_spectrum: "Spectrum Forecast",
  megapixels_target: "Megapixels",
  length_frames: "Duration (frames)",
  audio_frames_override: "Audio Length Override",
  video_frames_override: "Video Length Override",
  vae_decode_tiled: "VAE Decode (Tiled)",
  unload_before_decode: "Unload \u2192 VAE",
  low_vram: "Low VRAM",
  generated_audio_gain_db: "Audio Gain (dB)",
  enable_derope: "De-ROPE",
  derope_preset: "Preset",
  derope_inject: "Inject Strength",
  derope_audio: "Audio Source",
  derope_mode: "Dilation Mode",
  derope_profile: "Oracle Profile",
  derope_abstain: "Abstain Below",
  derope_protect_tail: "Protect Tail",
  enable_derope_adapter: "Motion Adapter",
  enable_negpip: "NegPiP",
};

// Explicit +/- step per widget (units per click; hold-to-repeat uses the same).
const NUMBER_STEPS = {
  steps: 1,
  steps_turbo: 1,
  shift_video: 0.5,
  shift_audio: 0.5,
  megapixels_target: 0.1,
  length_frames: 17,
  generated_audio_gain_db: 0.1,
  derope_inject: 0.05,
  derope_abstain: 0.1,
  derope_protect_tail: 1,
};

// Value snapping applied on commit and after each step.
const NUMBER_BEHAVIOUR = {
  length_frames: { snap: snapFrameCount },
};

function snapFrameCount(value) {
  let f = Math.round(Number(value));
  if (!Number.isFinite(f)) f = 124;
  if (f <= 5) return 5;
  const n = Math.ceil((f - 5) / 17);
  return n * 17 + 5;
}

function log(...args) {
  if (window.__crtH3USDebug) console.log("[CRT H3US]", ...args);
}

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function getComboOptions(widget) {
  const options = widget?.options?.values;
  if (Array.isArray(options)) return options.map((v) => String(v));
  if (options instanceof Function) {
    try {
      return options().map((v) => String(v));
    } catch {
      return [];
    }
  }
  return [];
}

// Sub-knobs with progressive disclosure. Each field maps to a list of
// conditions that must ALL hold: a string means "master is truthy", an object
// means "master equals value". The de-rope tuning knobs only appear when the
// preset is Custom, so the section stays a single dropdown by default.
const SUB_FIELD_MASTERS = {
  derope_preset: ["enable_derope"],
  enable_derope_adapter: ["enable_derope"],
  derope_inject: ["enable_derope", { master: "derope_preset", value: "Custom" }],
  derope_audio: ["enable_derope", { master: "derope_preset", value: "Custom" }],
  derope_mode: ["enable_derope", { master: "derope_preset", value: "Custom" }],
  derope_profile: ["enable_derope", { master: "derope_preset", value: "Custom" }],
  derope_abstain: ["enable_derope", { master: "derope_preset", value: "Custom" }],
  derope_protect_tail: ["enable_derope", { master: "derope_preset", value: "Custom" }],
};

function fieldLabel(name) {
  return FIELD_LABELS[name] || name;
}

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;

  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .crt-h3us-root {
      width: 100%;
      height: 0;
      box-sizing: border-box;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
      user-select: none;
      -webkit-user-select: none;
      pointer-events: none;
    }

    .crt-h3us-shell {
      --bg-base: #09090b;
      --bg-surface: #111114;
      --bg-elevated: #18181b;
      --bg-hover: #1f1f23;
      --border-subtle: rgba(255, 255, 255, 0.04);
      --border-default: rgba(255, 255, 255, 0.08);
      --text-primary: #fafafa;
      --text-secondary: #71717a;
      --text-tertiary: #52525b;
      --accent: #22d3ee;
      --accent-soft: rgba(34, 211, 238, 0.12);
      --accent-glow: rgba(34, 211, 238, 0.25);
      --success: #22c55e;

      width: calc(100% - 12px);
      margin: 6px;
      padding: 10px;
      border-radius: 12px;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      color: var(--text-primary);
      box-sizing: border-box;
      pointer-events: auto;
      position: relative;
      overflow: visible;
    }

    .crt-h3us-shell::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(34, 211, 238, 0.3), transparent);
    }

    .crt-h3us-shell * {
      box-sizing: border-box;
      pointer-events: auto;
      user-select: none;
      -webkit-user-select: none;
    }

    .crt-h3us-title {
      font-size: 11px;
      font-weight: 500;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--text-tertiary);
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .crt-h3us-title::before {
      content: '';
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 8px var(--accent-glow);
    }

    .crt-h3us-tabs {
      display: flex;
      gap: 2px;
      padding: 3px;
      background: var(--bg-base);
      border-radius: 8px;
      border: 1px solid var(--border-subtle);
      justify-content: center;
      width: fit-content;
      margin: 0 auto 8px auto;
    }

    .crt-h3us-tab {
      height: 24px;
      padding: 0 10px;
      border-radius: 6px;
      border: none;
      background: transparent;
      color: var(--text-tertiary);
      font-size: 11px;
      font-weight: 500;
      cursor: pointer;
      transition: all 180ms ease;
      white-space: nowrap;
    }

    .crt-h3us-tab:hover {
      color: var(--text-secondary);
    }

    .crt-h3us-tab.mode-active {
      background: var(--bg-elevated);
      color: var(--accent);
      border: 1px solid rgba(34, 211, 238, 0.55);
      box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.35), 0 0 16px rgba(34, 211, 238, 0.32);
    }

    .crt-h3us-tab.view-active:not(.mode-active) {
      background: var(--bg-elevated);
      color: var(--text-primary);
    }

    .crt-h3us-tab.settings {
      color: var(--accent);
    }

    .crt-h3us-tab.preview {
      color: var(--text-tertiary);
    }

    .crt-h3us-tab.preview.is-hidden {
      display: none;
    }

    .crt-h3us-tab.preview.generating {
      color: var(--success);
      text-shadow: 0 0 10px rgba(34, 197, 94, 0.6);
    }

    .crt-h3us-panel {
      display: none;
      flex-direction: column;
      gap: 6px;
    }

    .crt-h3us-panel.active {
      display: flex;
    }

    .crt-h3us-hint {
      font-size: 10.5px;
      line-height: 1.45;
      color: var(--text-secondary);
      background: var(--bg-base);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 7px 9px;
    }

    .crt-h3us-section {
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      background: var(--bg-base);
      overflow: hidden;
    }

    .crt-h3us-sec-head {
      display: flex;
      align-items: center;
      gap: 7px;
      width: 100%;
      padding: 6px 9px;
      border: none;
      background: transparent;
      cursor: pointer;
      text-align: left;
    }

    .crt-h3us-sec-head:hover {
      background: var(--bg-hover);
    }

    .crt-h3us-arrow {
      color: var(--text-tertiary);
      font-size: 10px;
      transition: transform 160ms ease;
      width: 10px;
      flex: 0 0 10px;
    }

    .crt-h3us-section.open .crt-h3us-arrow {
      transform: rotate(90deg);
      color: var(--accent);
    }

    .crt-h3us-sec-title {
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.07em;
      text-transform: uppercase;
      color: var(--text-secondary);
    }

    .crt-h3us-section.open .crt-h3us-sec-title {
      color: var(--text-primary);
    }

    .crt-h3us-sec-body {
      display: none;
      flex-direction: column;
      gap: 6px;
      padding: 4px 9px 8px 9px;
    }

    .crt-h3us-section.open .crt-h3us-sec-body {
      display: flex;
    }

    .crt-h3us-field {
      display: grid;
      grid-template-columns: 118px 1fr;
      align-items: center;
      gap: 8px;
      min-height: 26px;
    }

    .crt-h3us-label {
      font-size: 11px;
      color: var(--text-secondary);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .crt-h3us-control {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      min-width: 0;
    }

    .crt-h3us-select-wrap {
      position: relative;
      width: 100%;
    }

    .crt-h3us-select {
      width: 100%;
      height: 26px;
      padding: 0 8px;
      border-radius: 7px;
      border: 1px solid var(--border-default);
      background: var(--bg-base);
      color: var(--text-primary);
      font-size: 11px;
      appearance: none;
      cursor: pointer;
    }

    .crt-h3us-select:focus {
      outline: none;
      border-color: rgba(34, 211, 238, 0.5);
    }

    .crt-h3us-num {
      display: flex;
      align-items: center;
      gap: 3px;
      width: 100%;
      justify-content: flex-end;
    }

    .crt-h3us-num-btn {
      width: 22px;
      height: 22px;
      flex: 0 0 22px;
      border-radius: 6px;
      border: 1px solid var(--border-default);
      background: var(--bg-elevated);
      color: var(--text-secondary);
      font-size: 12px;
      line-height: 1;
      cursor: pointer;
    }

    .crt-h3us-num-btn:hover {
      background: var(--bg-hover);
      color: var(--text-primary);
    }

    .crt-h3us-num-input {
      width: 64px;
      height: 24px;
      text-align: center;
      border-radius: 6px;
      border: 1px solid var(--border-default);
      background: var(--bg-base);
      color: var(--text-primary);
      font-size: 11px;
    }

    .crt-h3us-num-input:focus {
      outline: none;
      border-color: rgba(34, 211, 238, 0.5);
    }

    .crt-h3us-bool {
      display: flex;
      align-items: center;
      gap: 7px;
      cursor: pointer;
      justify-content: flex-end;
      width: 100%;
    }

    .crt-h3us-bool-text {
      font-size: 11px;
      color: var(--text-tertiary);
    }

    .crt-h3us-toggle {
      width: 30px;
      height: 16px;
      border-radius: 999px;
      background: var(--bg-base);
      border: 1px solid var(--border-default);
      position: relative;
      transition: all 160ms ease;
      flex: 0 0 30px;
    }

    .crt-h3us-toggle::after {
      content: '';
      position: absolute;
      top: 1.5px;
      left: 2px;
      width: 11px;
      height: 11px;
      border-radius: 50%;
      background: var(--text-tertiary);
      transition: all 160ms ease;
    }

    .crt-h3us-toggle.on {
      background: var(--accent-soft);
      border-color: rgba(34, 211, 238, 0.55);
    }

    .crt-h3us-toggle.on::after {
      left: 15px;
      background: var(--accent);
      box-shadow: 0 0 8px var(--accent-glow);
    }

    .crt-h3us-preview {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 6px;
    }

    .crt-h3us-preview-status {
      font-size: 10.5px;
      color: var(--text-tertiary);
    }

    .crt-h3us-preview-status.live {
      color: var(--success);
    }

    .crt-h3us-preview-image {
      display: none;
      max-width: 100%;
      max-height: 260px;
      border-radius: 8px;
      border: 1px solid var(--border-subtle);
    }
  `;
  document.head.appendChild(style);
}

function parseNumber(rawValue, widget) {
  let value = Number(String(rawValue).replace(",", "."));
  if (!Number.isFinite(value)) value = Number(widget?.options?.default ?? 0);
  if (widget?.options) {
    if (typeof widget.options.min === "number") value = Math.max(widget.options.min, value);
    if (typeof widget.options.max === "number") value = Math.min(widget.options.max, value);
  }
  return value;
}

class MiniMaxH3UnifiedSamplerUI {
  constructor(node) {
    this.node = node;
    this.controls = new Map();
    this.panels = new Map();
    this.tabs = new Map();
    // Expand-on-toggle sub-rows: hidden until their master toggle is on.
    this.subRows = [];
    this.resizeTimer = null;
    this.previewUrl = null;
    this.previewHandler = null;
    this.previewMetaHandler = null;
    this.cleanups = [];
    this.generationActive = false;
    this.generationHandlers = null;
    this.executingNodeId = null;

    const modeWidget = getWidget(node, "workflow_mode");
    this.mode = WORKFLOW_MODES.includes(String(modeWidget?.value)) ? String(modeWidget.value) : WORKFLOW_MODES[0];
    const saved = this.node.properties?.h3us_view;
    this.activeView = [...WORKFLOW_MODES, "SETTINGS", "PREVIEW"].includes(saved) ? saved : this.mode;
    // Mode tabs are not independent views — never keep a stale mode view
    // (e.g. I2V) when the workflow_mode is T2V. Settings/Preview are kept.
    if (WORKFLOW_MODES.includes(this.activeView) && this.activeView !== this.mode) {
      this.activeView = this.mode;
      try { this.node.properties.h3us_view = this.mode; } catch {}
    }

    this.init();
  }

  init() {
    ensureStyles();
    this.hideNativeWidgets();
    this.createContainer();
    this.buildLayout();
    this.syncFromWidgets();
    this.refresh();
    this.bindPreview();
    this.bindGeneration();
    this.scheduleResize();
  }

  hideNativeWidgets() {
    for (const widget of this.node.widgets || []) {
      if (widget.name === "h3us_ui") continue;
      widget.hidden = true;
      widget.computeSize = () => [0, -6];
    }
  }

  createContainer() {
    if (this.container) return;
    this.container = document.createElement("div");
    this.container.className = "crt-h3us-root";
    this.domWidget = this.node.addDOMWidget("h3us_ui", "div", this.container, {
      serialize: false,
    });
    if (this.domWidget) {
      this.domWidget.computeSize = () => [0, 0];
    }
    this.syncDOMHitbox();
  }

  syncDOMHitbox() {
    const parent = this.container?.parentElement;
    const elements = [this.domWidget?.element, this.container];
    if (parent && parent !== document.body && parent.children?.length === 1) {
      elements.push(parent);
    }
    for (const element of elements) {
      if (!element?.style) continue;
      element.style.pointerEvents = "none";
      element.style.overflow = "visible";
    }
    if (this.container?.style) {
      this.container.style.height = "0px";
    }
  }

  buildLayout() {
    this.container.innerHTML = "";
    this.controls.clear();
    this.panels.clear();
    this.tabs.clear();
    this.subRows = [];

    const shell = document.createElement("div");
    shell.className = "crt-h3us-shell";
    this.shell = shell;

    const title = document.createElement("div");
    title.className = "crt-h3us-title";
    title.textContent = "MiniMax H3";
    shell.appendChild(title);

    const tabsWrap = document.createElement("div");
    tabsWrap.className = "crt-h3us-tabs";
    shell.appendChild(tabsWrap);
    this.tabsWrap = tabsWrap;

    for (const mode of WORKFLOW_MODES) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "crt-h3us-tab";
      button.textContent = mode;
      button.addEventListener("click", () => this.setMode(mode));
      tabsWrap.appendChild(button);
      this.tabs.set(mode, button);
    }

    const settingsButton = document.createElement("button");
    settingsButton.type = "button";
    settingsButton.className = "crt-h3us-tab settings";
    settingsButton.textContent = "Settings";
    settingsButton.addEventListener("click", () => {
      this.activeView = "SETTINGS";
      this.persistView();
      this.refresh();
    });
    tabsWrap.appendChild(settingsButton);
    this.tabs.set("SETTINGS", settingsButton);

    const previewButton = document.createElement("button");
    previewButton.type = "button";
    previewButton.className = "crt-h3us-tab preview";
    previewButton.textContent = "Preview";
    previewButton.addEventListener("click", () => {
      this.activeView = "PREVIEW";
      this.persistView();
      this.ensureTAESDPreviewMethod();
      this.refresh();
    });
    tabsWrap.appendChild(previewButton);
    this.tabs.set("PREVIEW", previewButton);

    this.panelHost = document.createElement("div");
    shell.appendChild(this.panelHost);

    this.buildPanels();
    this.container.appendChild(shell);
    this.syncDOMHitbox();
  }

  buildPanels() {
    this.panelHost.innerHTML = "";
    this.panels.clear();

    for (const mode of WORKFLOW_MODES) {
      const panel = document.createElement("div");
      panel.className = "crt-h3us-panel";

      for (const name of MODE_FIELDS[mode] || []) {
        const row = this.buildFieldRow(name);
        if (row) panel.appendChild(row);
      }

      this.panelHost.appendChild(panel);
      this.panels.set(mode, panel);
    }

    const settingsPanel = document.createElement("div");
    settingsPanel.className = "crt-h3us-panel";
    this.sectionBodies = {};

    for (const section of SETTINGS_SECTIONS) {
      const secEl = document.createElement("div");
      secEl.className = "crt-h3us-section";

      const head = document.createElement("button");
      head.type = "button";
      head.className = "crt-h3us-sec-head";

      const arrow = document.createElement("span");
      arrow.className = "crt-h3us-arrow";
      arrow.textContent = "\u25B8";

      const secTitle = document.createElement("span");
      secTitle.className = "crt-h3us-sec-title";
      secTitle.textContent = section.title;

      head.appendChild(arrow);
      head.appendChild(secTitle);

      const body = document.createElement("div");
      body.className = "crt-h3us-sec-body";

      if (section.kind === "preview-row") {
        const row = this.buildLivePreviewRow();
        if (row) body.appendChild(row);
      } else {
        for (const name of section.fields || []) {
          const row = this.buildFieldRow(name);
          if (row) body.appendChild(row);
        }
      }

      head.addEventListener("click", () => {
        const open = !secEl.classList.contains("open");
        secEl.classList.toggle("open", open);
        this.persistSectionState(section.id, open);
        this.scheduleResize();
      });

      secEl.appendChild(head);
      secEl.appendChild(body);
      settingsPanel.appendChild(secEl);
      this.sectionBodies[section.id] = { el: secEl, def: section };
    }

    this.panelHost.appendChild(settingsPanel);
    this.panels.set("SETTINGS", settingsPanel);

    const previewPanel = document.createElement("div");
    previewPanel.className = "crt-h3us-panel";
    previewPanel.appendChild(this.buildPreviewPanel());
    this.panelHost.appendChild(previewPanel);
    this.panels.set("PREVIEW", previewPanel);
  }

  persistSectionState(id, open) {
    this.node.properties ??= {};
    const state = this.node.properties.h3us_sections ??= {};
    if (open) {
      state[id] = true;
    } else {
      delete state[id];
    }
  }

  restoreSectionState() {
    const state = this.node.properties?.h3us_sections || {};
    for (const [id, entry] of Object.entries(this.sectionBodies || {})) {
      entry.el.classList.toggle("open", Boolean(state[id]));
    }
  }

  registerControl(name, entry) {
    // The same widget can be rendered on more than one panel (aspect_ratio on
    // T2V and R2V); keep every live element so syncing updates them all.
    const list = this.controls.get(name) || [];
    list.push(entry);
    this.controls.set(name, list);
  }

  buildPreviewPanel() {
    const wrap = document.createElement("div");
    wrap.className = "crt-h3us-preview";

    this.previewStatus = document.createElement("div");
    this.previewStatus.className = "crt-h3us-preview-status";
    wrap.appendChild(this.previewStatus);

    this.previewImage = document.createElement("img");
    this.previewImage.className = "crt-h3us-preview-image";
    this.previewImage.alt = "Preview";
    wrap.appendChild(this.previewImage);

    this.updatePreviewStatus();
    return wrap;
  }

  buildFieldRow(name) {
    const widget = getWidget(this.node, name);
    if (!widget) return null;

    const row = document.createElement("div");
    row.className = "crt-h3us-field";
    const tooltip = widget?.options?.tooltip;
    if (tooltip) {
      row.title = String(tooltip);
    }

    const label = document.createElement("div");
    label.className = "crt-h3us-label";
    label.textContent = fieldLabel(name);
    row.appendChild(label);

    const controlWrap = document.createElement("div");
    controlWrap.className = "crt-h3us-control";
    row.appendChild(controlWrap);

    const options = getComboOptions(widget);
    const hasNumericRange = Boolean(
      widget?.options &&
        (typeof widget.options.min === "number" ||
          typeof widget.options.max === "number" ||
          typeof widget.options.step === "number"),
    );
    const isBool = typeof widget.value === "boolean" || widget.type === "toggle";
    const isCombo = options.length > 0 || widget.type === "combo";

    if (hasNumericRange && !isBool) {
      controlWrap.appendChild(this.makeNumber(name, widget));
    } else if (isBool) {
      controlWrap.appendChild(this.makeBool(name, widget));
    } else if (isCombo) {
      const selectWrap = document.createElement("div");
      selectWrap.className = "crt-h3us-select-wrap";
      selectWrap.appendChild(this.makeCombo(name, widget));
      controlWrap.appendChild(selectWrap);
    } else if (typeof widget.value === "number") {
      controlWrap.appendChild(this.makeNumber(name, widget));
    }

    const conditions = SUB_FIELD_MASTERS[name];
    if (conditions) {
      row.dataset.subOf = conditions
        .map((c) => (typeof c === "string" ? c : c.master))
        .join(",");
      this.subRows.push({ row, conditions });
    }

    return row;
  }

  makeCombo(name, widget) {
    const select = document.createElement("select");
    select.className = "crt-h3us-select";

    for (const optionValue of getComboOptions(widget)) {
      const option = document.createElement("option");
      option.value = String(optionValue);
      option.textContent = String(optionValue);
      select.appendChild(option);
    }

    select.value = String(widget.value ?? "");
    select.addEventListener("change", () => this.writeWidget(name, widget, select.value));
    this.registerControl(name, { kind: "combo", element: select, widget });
    return select;
  }

  makeBool(name, widget) {
    const root = document.createElement("label");
    root.className = "crt-h3us-bool";

    const text = document.createElement("span");
    text.className = "crt-h3us-bool-text";
    text.textContent = Boolean(widget.value) ? "Enabled" : "Disabled";

    const hidden = document.createElement("input");
    hidden.type = "checkbox";
    hidden.style.display = "none";
    hidden.checked = Boolean(widget.value);

    const toggle = document.createElement("span");
    toggle.className = "crt-h3us-toggle";
    if (hidden.checked) toggle.classList.add("on");

    hidden.addEventListener("change", () => {
      const checked = hidden.checked;
      text.textContent = checked ? "Enabled" : "Disabled";
      toggle.classList.toggle("on", checked);
      this.writeWidget(name, widget, checked);
    });

    root.addEventListener("click", (event) => {
      event.preventDefault();
      hidden.checked = !hidden.checked;
      hidden.dispatchEvent(new Event("change"));
    });

    root.appendChild(text);
    root.appendChild(hidden);
    root.appendChild(toggle);

    this.registerControl(name, {
      kind: "bool",
      element: hidden,
      label: text,
      toggle,
      widget,
    });
    return root;
  }

  makeNumber(name, widget) {
    const wrap = document.createElement("div");
    wrap.className = "crt-h3us-num";
    const behaviour = NUMBER_BEHAVIOUR[name];

    const minus = document.createElement("button");
    minus.type = "button";
    minus.className = "crt-h3us-num-btn";
    minus.textContent = "-";

    const input = document.createElement("input");
    input.type = "text";
    input.className = "crt-h3us-num-input";
    const defaultNumber = typeof widget?.options?.default === "number" ? widget.options.default : 0;
    const initialNumber = typeof widget.value === "number" ? widget.value : defaultNumber;
    input.value = behaviour ? behaviour.snap(initialNumber) : initialNumber;

    const plus = document.createElement("button");
    plus.type = "button";
    plus.className = "crt-h3us-num-btn";
    plus.textContent = "+";

    const min = typeof widget?.options?.min === "number" ? widget.options.min : null;
    const max = typeof widget?.options?.max === "number" ? widget.options.max : null;
    const stepSize = NUMBER_STEPS[name] ?? 1;
    const isIntStep = Number.isInteger(stepSize);

    const clampValue = (value) => {
      let next = value;
      if (min !== null) next = Math.max(min, next);
      if (max !== null) next = Math.min(max, next);
      return next;
    };

    const commit = (rawValue) => {
      let parsed = parseNumber(rawValue, widget);
      if (behaviour) {
        parsed = clampValue(parsed);
        parsed = behaviour.snap(parsed);
      }
      input.value = parsed;
      this.writeWidget(name, widget, parsed);
      return parsed;
    };

    const applyDelta = (direction) => {
      let current = Number(widget.value);
      if (!Number.isFinite(current)) current = Number(input.value);
      if (!Number.isFinite(current)) current = defaultNumber;
      let next = clampValue(current + direction * stepSize);
      if (behaviour) {
        next = behaviour.snap(next);
      } else if (isIntStep) {
        next = Math.round(next);
      } else {
        next = Number(next.toFixed(4));
      }
      input.value = next;
      this.writeWidget(name, widget, next);
    };

    const startHold = (direction) => {
      applyDelta(direction);
      let repeatId = null;
      const kickoffId = window.setTimeout(() => {
        repeatId = window.setInterval(() => applyDelta(direction), 100);
      }, 500);
      const stop = () => {
        window.clearTimeout(kickoffId);
        if (repeatId !== null) window.clearInterval(repeatId);
        window.removeEventListener("mouseup", stop);
      };
      window.addEventListener("mouseup", stop, { once: true });
    };

    minus.addEventListener("mousedown", (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      startHold(-1);
    });

    plus.addEventListener("mousedown", (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      startHold(1);
    });

    input.addEventListener("blur", () => commit(input.value));

    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        commit(input.value);
        input.blur();
      }
      if (event.key === "Escape") {
        input.value = widget.value ?? "";
        input.blur();
      }
    });

    wrap.appendChild(minus);
    wrap.appendChild(input);
    wrap.appendChild(plus);

        this.registerControl(name, {
      kind: "number",
      element: input,
      minus,
      plus,
      widget,
    });
    return wrap;
  }

  writeWidget(name, widget, value) {
    const previous = widget.value;
    widget.value = value;
    if (typeof widget.callback === "function") {
      try {
        widget.callback(value, app.canvas, this.node, undefined, widget);
      } catch (error) {
        log("widget callback failed", name, error);
      }
    }
    if (previous !== value) {
      this.node.setDirtyCanvas(true, true);
    }
    if (name === "enable_derope" || name === "enable_derope_adapter" || name === "derope_preset" || name === "enable_negpip") {
      this.updateSubRowVisibility();
      this.scheduleResize();
    }
    if (name === "workflow_mode") {
      this.mode = String(value);
      if (this.activeView === previous) {
        this.activeView = this.mode;
      }
      // Immediate name-keyed persist - survives F5 even before the next
      // serialize (which needs the new JS to have run once).
      try {
        this.node.properties ??= {};
        this.node.properties.h3us_mode = String(value);
        this.node.properties.h3us_widgets ??= {};
        this.node.properties.h3us_widgets.workflow_mode = String(value);
      } catch {}
      this.persistView();
      this.refresh();
    }
  }

  setMode(mode) {
    const modeWidget = getWidget(this.node, "workflow_mode");
    this.mode = mode;
    if (modeWidget) {
      this.writeWidget("workflow_mode", modeWidget, mode);
    }
    this.activeView = mode;
    this.persistView();
    this.refresh();
  }

  clearPreview() {
    if (this.previewUrl) {
      try { URL.revokeObjectURL(this.previewUrl); } catch {}
      this.previewUrl = null;
    }
    if (this.previewImage) {
      this.previewImage.removeAttribute("src");
      this.previewImage.style.display = "none";
    }
  }

  buildLivePreviewRow() {
    const widget = getWidget(this.node, "live_preview");
    if (!widget) return null;

    const row = document.createElement("div");
    row.className = "crt-h3us-field";

    const label = document.createElement("div");
    label.className = "crt-h3us-label";
    label.textContent = "Live Preview";
    row.appendChild(label);

    const controlWrap = document.createElement("div");
    controlWrap.className = "crt-h3us-control";
    row.appendChild(controlWrap);

    const root = document.createElement("label");
    root.className = "crt-h3us-bool";

    const text = document.createElement("span");
    text.className = "crt-h3us-bool-text";
    text.textContent = Boolean(widget.value) ? "Enabled" : "Disabled";

    const hidden = document.createElement("input");
    hidden.type = "checkbox";
    hidden.style.display = "none";
    hidden.checked = Boolean(widget.value);

    const toggle = document.createElement("span");
    toggle.className = "crt-h3us-toggle";
    if (hidden.checked) toggle.classList.add("on");

    hidden.addEventListener("change", () => {
      const checked = hidden.checked;
      text.textContent = checked ? "Enabled" : "Disabled";
      toggle.classList.toggle("on", checked);
      this.writeWidget("live_preview", widget, checked);
      this.refresh();
    });

    root.addEventListener("click", (event) => {
      event.preventDefault();
      hidden.checked = !hidden.checked;
      hidden.dispatchEvent(new Event("change"));
    });

    root.appendChild(text);
    root.appendChild(hidden);
    root.appendChild(toggle);
    controlWrap.appendChild(root);

    this.livePreviewControl = { element: hidden, label: text, toggle, widget };
    return row;
  }

  syncFromWidgets() {
    const modeWidget = getWidget(this.node, "workflow_mode");
    if (modeWidget) {
      this.mode = String(modeWidget.value);
      if (!WORKFLOW_MODES.includes(this.mode)) this.mode = WORKFLOW_MODES[0];
    }
    for (const [name, controlList] of this.controls.entries()) {
      const widget = getWidget(this.node, name);
      if (!widget) continue;
      const behaviour = NUMBER_BEHAVIOUR[name];
      for (const control of controlList) {
        control.widget = widget;
        if (control.kind === "bool") {
          const checked = Boolean(widget.value);
          control.element.checked = checked;
          control.label.textContent = checked ? "Enabled" : "Disabled";
          control.toggle.classList.toggle("on", checked);
        } else if (control.kind === "number") {
          const shown = behaviour ? behaviour.snap(widget.value) : widget.value;
          control.element.value = String(shown);
        } else {
          control.element.value = String(widget.value ?? "");
        }
      }
    }
  }

  persistView() {
    this.node.properties ??= {};
    if (["SETTINGS", "PREVIEW", ...WORKFLOW_MODES].includes(this.activeView)) {
      this.node.properties.h3us_view = this.activeView;
    } else {
      delete this.node.properties.h3us_view;
    }
  }

  refresh() {
    this.syncFromWidgets();
    // Mode tabs *are* the mode — never display an I2V panel when mode is T2V.
    if (WORKFLOW_MODES.includes(this.activeView) && this.activeView !== this.mode) {
      this.activeView = this.mode;
      this.persistView();
    }
    if (!["SETTINGS", "PREVIEW", ...WORKFLOW_MODES].includes(this.activeView)) {
      this.activeView = this.mode;
    }
    this.updatePreviewTabVisibility();
    this.updateSubRowVisibility();
    for (const [key, button] of this.tabs.entries()) {
      button.classList.toggle("mode-active", WORKFLOW_MODES.includes(key) && key === this.mode);
      button.classList.toggle("view-active", key === this.activeView);
    }
    if (!this.panels.has(this.activeView)) {
      this.activeView = this.mode;
      this.persistView();
    }
    for (const [key, panel] of this.panels.entries()) {
      panel.classList.toggle("active", key === this.activeView);
    }
    this.updateLivePreviewControl();
    if (this.activeView === "PREVIEW") {
      this.updatePreviewStatus();
    }
    this.scheduleResize();
  }

  rebuild() {
    this.hideNativeWidgets();
    this.controls.clear();
    this.buildPanels();
    this.restoreSectionState();
    this.syncFromWidgets();
    this.refresh();
  }

  isLivePreviewEnabled() {
    const widget = getWidget(this.node, "live_preview");
    return Boolean(widget?.value);
  }

  updatePreviewTabVisibility() {
    const previewTab = this.tabs.get("PREVIEW");
    if (!previewTab) return;
    const enabled = this.isLivePreviewEnabled();
    previewTab.classList.toggle("is-hidden", !enabled);
    if (!enabled && this.activeView === "PREVIEW") {
      this.activeView = this.mode;
      this.persistView();
    }
  }

  updateSubRowVisibility() {
    for (const entry of this.subRows || []) {
      const enabled = (entry.conditions || []).every((c) => {
        const widget = getWidget(this.node, typeof c === "string" ? c : c.master);
        if (!widget) return false;
        return c.value === undefined
          ? Boolean(widget.value)
          : String(widget.value) === String(c.value);
      });
      entry.row.style.display = enabled ? "" : "none";
    }
  }

  updateLivePreviewControl() {
    if (!this.livePreviewControl) return;
    const widget = getWidget(this.node, "live_preview");
    if (!widget) return;
    const checked = Boolean(widget.value);
    this.livePreviewControl.element.checked = checked;
    this.livePreviewControl.label.textContent = checked ? "Enabled" : "Disabled";
    this.livePreviewControl.toggle.classList.toggle("on", checked);
  }

  isTAESDActive() {
    try {
      const value =
        app.ui?.settings?.getSettingValue?.("Comfy.PreviewMethod") ??
        app.ui?.settings?.getSettingValue?.("preview_method") ??
        "";
      return String(value).toLowerCase().includes("taesd");
    } catch {
      return false;
    }
  }

  ensureTAESDPreviewMethod() {
    if (!this.isLivePreviewEnabled()) return;
    if (this.isTAESDActive()) return;
    try {
      const settings = app.ui?.settings;
      if (typeof settings?.setSettingValue === "function") {
        settings.setSettingValue("Comfy.PreviewMethod", "taesd");
        settings.setSettingValue("preview_method", "taesd");
      }
    } catch {
      // Ignore
    }
  }

  updatePreviewStatus() {
    if (!this.previewStatus) return;
    if (!this.isLivePreviewEnabled()) {
      this.previewStatus.classList.remove("live");
      this.previewStatus.textContent = "Live preview inactive";
      return;
    }
    this.ensureTAESDPreviewMethod();
    const live = this.isTAESDActive();
    this.previewStatus.classList.toggle("live", live);
    this.previewStatus.textContent = live
      ? "Live preview active (taeh3 / RGB factors)"
      : "Enable TAESD in settings for live preview";
  }

  bindPreview() {
    const applyPreviewBlob = (blob) => {
      if (!(blob instanceof Blob) || !this.previewImage) return;
      const url = URL.createObjectURL(blob);
      if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
      this.previewUrl = url;
      this.previewImage.src = url;
      this.previewImage.style.display = "block";
    };

    const applyPreviewHTMLImage = (img) => {
      if (!this.previewImage || !(img instanceof HTMLImageElement) || !img.src) return;
      if (this.previewImage.src === img.src) return;
      this.previewImage.src = img.src;
      this.previewImage.style.display = "block";
    };

    this.previewMetaHandler = (event) => {
      const payload = event.detail;
      if (!payload) return;
      const nodeId = payload.nodeId ?? payload.node_id ?? payload.displayNodeId;
      if (nodeId !== undefined && String(nodeId) !== String(this.node.id)) return;
      if (payload instanceof Blob) {
        applyPreviewBlob(payload);
      } else if (payload.blob instanceof Blob) {
        applyPreviewBlob(payload.blob);
      } else if (payload.image instanceof Blob) {
        applyPreviewBlob(payload.image);
      }
    };

    // Untagged core previews (one per sampler step) arrive as bare blobs; show
    // them only while this node is the one executing.
    this.previewStepHandler = (event) => {
      if (!(event.detail instanceof Blob)) return;
      if (String(this.executingNodeId ?? "") !== String(this.node.id)) return;
      applyPreviewBlob(event.detail);
    };

    api.addEventListener("b_preview_with_metadata", this.previewMetaHandler);
    api.addEventListener("b_preview", this.previewStepHandler);
    this._bindNodeImgs(applyPreviewHTMLImage);
  }

  _bindNodeImgs(onImage) {
    const node = this.node;
    if (node._h3usImgsIntercepted) return;
    node._h3usImgsIntercepted = true;
    let _imgs;
    Object.defineProperty(node, "imgs", {
      get() {
        return _imgs;
      },
      set(value) {
        if (Array.isArray(value) && value.length > 0) {
          const src = value[0]?.src ?? "";
          if (src.startsWith("blob:")) {
            onImage(value[0]);
            return;
          }
          _imgs = value;
          onImage(value[0]);
          return;
        }
        _imgs = value;
      },
      configurable: true,
      enumerable: true,
    });
  }

  _unbindNodeImgs() {
    const node = this.node;
    if (!node?._h3usImgsIntercepted) return;
    const current = node.imgs;
    delete node._h3usImgsIntercepted;
    try {
      Object.defineProperty(node, "imgs", {
        value: current,
        writable: true,
        configurable: true,
        enumerable: true,
      });
    } catch {
      // Ignore
    }
  }

  bindGeneration() {
    if (this.generationHandlers) return;

    const onStart = () => {
      this.clearPreview();
      this.setGenerating(true);
    };
    const onExecuting = ({ detail }) => {
      // The detail is the executing node id (or an object with node fields in
      // some frontends); null means execution ended.
      let nodeId = detail;
      if (nodeId && typeof nodeId === "object") {
        nodeId = nodeId.node ?? nodeId.display_node ?? null;
      }
      this.executingNodeId = nodeId ?? null;
      // New run of *this* node - drop the stale image from the previous run.
      if (String(nodeId ?? "") === String(this.node.id)) {
        this.clearPreview();
      }
      if (detail === null) this.setGenerating(false);
    };
    const onStop = () => this.setGenerating(false);

    this.generationHandlers = { onStart, onExecuting, onStop };
    api.addEventListener("execution_start", onStart);
    api.addEventListener("executing", onExecuting);
    api.addEventListener("execution_error", onStop);
    api.addEventListener("execution_interrupted", onStop);
  }

  setGenerating(active) {
    const next = Boolean(active);
    if (this.generationActive === next) return;
    this.generationActive = next;
    const previewTab = this.tabs.get("PREVIEW");
    previewTab?.classList.toggle("generating", this.generationActive);
  }

  scheduleResize() {
    window.clearTimeout(this.resizeTimer);
    this.resizeTimer = window.setTimeout(() => {
      this.resizeTimer = null;
      this.updateSize();
    }, 36);
  }

  updateSize() {
    this.syncDOMHitbox();
    const targetWidth = MIN_WIDTH;
    const targetHeight =
      typeof this.node._h3usCompactHeight === "function"
        ? this.node._h3usCompactHeight()
        : MIN_HEIGHT;
    if (this.node.size?.[0] !== targetWidth || this.node.size?.[1] !== targetHeight) {
      this.node.size = [targetWidth, targetHeight];
      this.node.setDirtyCanvas(true, true);
    }
  }

  destroy() {
    this._unbindNodeImgs();
    window.clearTimeout(this.resizeTimer);
    this.resizeTimer = null;
    for (const cleanup of this.cleanups) {
      try {
        cleanup();
      } catch {
        // Ignore
      }
    }
    this.cleanups = [];
    if (this.previewMetaHandler) {
      api.removeEventListener("b_preview_with_metadata", this.previewMetaHandler);
      this.previewMetaHandler = null;
    }
    if (this.previewStepHandler) {
      api.removeEventListener("b_preview", this.previewStepHandler);
      this.previewStepHandler = null;
    }
    if (this.generationHandlers) {
      api.removeEventListener("execution_start", this.generationHandlers.onStart);
      api.removeEventListener("executing", this.generationHandlers.onExecuting);
      api.removeEventListener("execution_error", this.generationHandlers.onStop);
      api.removeEventListener("execution_interrupted", this.generationHandlers.onStop);
      this.generationHandlers = null;
    }
    if (this.previewUrl) {
      URL.revokeObjectURL(this.previewUrl);
      this.previewUrl = null;
    }
    this.controls.clear();
    this.panels.clear();
    this.tabs.clear();
    this.container?.remove();
    this.domWidget = null;
    this.container = null;
    this.shell = null;
    this.previewImage = null;
    this.previewStatus = null;
  }
}

// Config uses native Autogrow — no custom JS needed.

app.registerExtension({
  name: "CRT.MiniMaxH3UnifiedSamplerUI",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const registeredName = String(nodeData?.name || "");
    const isSampler = registeredName === NODE_NAME || NODE_ALIASES.has(registeredName);
    const isConfig = registeredName === CONFIG_NODE_NAME;
    if (!isSampler && !isConfig) return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    const originalOnConfigure = nodeType.prototype.onConfigure;
    const originalOnSerialize = nodeType.prototype.onSerialize;
    const originalOnRemoved = nodeType.prototype.onRemoved;
    const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;

    // LiteGraph restores widget values positionally, which silently drops them
    // whenever the saved array no longer lines up with the live widgets list.
    // Mirror every scalar widget into properties by name and reapply on load.
    const snapshotWidgets = (node) => {
      try {
        const map = {};
        for (const widget of node.widgets || []) {
          if (!widget || widget.name === "h3us_ui") continue;
          if (widget.serialize === false) continue;
          const value = widget.value;
          if (
            typeof value === "number" ||
            typeof value === "boolean" ||
            typeof value === "string"
          ) {
            map[widget.name] = value;
          }
        }
        node.properties ??= {};
        node.properties.h3us_widgets = map;
      } catch {
        // Serialization must never fail because of the mirror.
      }
    };

    const restoreWidgets = (node) => {
      const saved = node.properties?.h3us_widgets;
      // Fallback for first F5 after the update (no h3us_widgets snapshot yet).
      const fallbackMode = node.properties?.h3us_mode;
      if (fallbackMode && WORKFLOW_MODES.includes(String(fallbackMode))) {
        const w = getWidget(node, "workflow_mode");
        if (w) w.value = String(fallbackMode);
      }
      if (!saved || typeof saved !== "object") return;
      for (const widget of node.widgets || []) {
        if (!widget || widget.type === "button" || widget.serialize === false) continue;
        if (!Object.prototype.hasOwnProperty.call(saved, widget.name)) continue;
        const value = saved[widget.name];
        if (
          typeof value === "number" ||
          typeof value === "boolean" ||
          typeof value === "string"
        ) {
          widget.value = value;
        }
      }
      // h3us_mode is written on every mode change — prefer it over the
      // snapshot if they disagree (snapshot may have cemented the default).
      if (fallbackMode && WORKFLOW_MODES.includes(String(fallbackMode))) {
        const w = getWidget(node, "workflow_mode");
        if (w) w.value = String(fallbackMode);
      }
    };

    if (isSampler) {
      const compactHeightForNode = (node) => {
        const probe = [0, 0];
        let maxVisibleY = Number(globalThis.LiteGraph?.NODE_TITLE_HEIGHT) || 30;

        const hasPosGetter = typeof node.getConnectionPos === "function";
        if (hasPosGetter) {
          if (Array.isArray(node.inputs)) {
            for (let i = 0; i < node.inputs.length; i++) {
              const slot = node.inputs[i];
              if (!slot || slot.hidden) continue;
              const pos = node.getConnectionPos(true, i, probe);
              const y = (Array.isArray(pos) ? pos[1] : probe[1]) - (node.pos?.[1] || 0);
              if (Number.isFinite(y)) maxVisibleY = Math.max(maxVisibleY, y);
            }
          }
          if (Array.isArray(node.outputs)) {
            for (let i = 0; i < node.outputs.length; i++) {
              const slot = node.outputs[i];
              if (!slot || slot.hidden) continue;
              const pos = node.getConnectionPos(false, i, probe);
              const y = (Array.isArray(pos) ? pos[1] : probe[1]) - (node.pos?.[1] || 0);
              if (Number.isFinite(y)) maxVisibleY = Math.max(maxVisibleY, y);
            }
          }
        }

        return Math.max(MIN_HEIGHT, Math.ceil(maxVisibleY + 8));
      };

      const clampNodeBounds = (node) => {
        const targetHeight = compactHeightForNode(node);
        if (node.size?.[0] !== MIN_WIDTH || node.size?.[1] !== targetHeight) {
          node.size = [MIN_WIDTH, targetHeight];
          node.setDirtyCanvas?.(true, false);
        }
        return targetHeight;
      };

      const applyNodeVisuals = (node) => {
        node.bgcolor = "transparent";
        node.color = "transparent";
        node.title = "";
        node.resizable = false;
        node.clip_area = false;
        node.flags ??= {};
        node.flags.clip_area = false;
        node._h3usCompactHeight = () => clampNodeBounds(node);

        if (!node._h3usVisualsPatched) {
          node._h3usVisualsPatched = true;
          node._h3usOriginalSetSize = node.setSize;
          node._h3usOriginalComputeSize = node.computeSize;
          node._h3usOriginalOnDrawForeground = node.onDrawForeground;
          node._h3usOriginalOnDrawBackground = node.onDrawBackground;

          node.computeSize = function (out) {
            const size = [MIN_WIDTH, compactHeightForNode(this)];
            if (out) {
              out[0] = size[0];
              out[1] = size[1];
              return out;
            }
            return size;
          };

          node.setSize = function () {
            const clamped = [MIN_WIDTH, compactHeightForNode(this)];
            this.size = clamped;
            return clamped;
          };

          node.onDrawBackground = function () {
            clampNodeBounds(this);
            return this._h3usOriginalOnDrawBackground?.apply(this, arguments);
          };

          node.onDrawForeground = function () {
            clampNodeBounds(this);
            return this._h3usOriginalOnDrawForeground?.apply(this, arguments);
          };
        }

        clampNodeBounds(node);
        node.setDirtyCanvas?.(true, true);
      };

      nodeType.prototype.onNodeCreated = function () {
        const result = originalOnNodeCreated?.apply(this, arguments);
        const workflowMode = getWidget(this, "workflow_mode");
        if (workflowMode && !WORKFLOW_MODES.includes(String(workflowMode.value))) {
          workflowMode.value = WORKFLOW_MODES[0];
        }
        this.properties ??= {};
        const viewValue = String(this.properties.h3us_view || "");
        if (!["SETTINGS", "PREVIEW", ...WORKFLOW_MODES].includes(viewValue)) {
          delete this.properties.h3us_view;
        }
        applyNodeVisuals(this);
        if (!this._h3usPovBuilt) {
          this._h3usPovBuilt = true;
          h3usBuildPreviewWidget(this);
        }
        if (!this.h3usUI) {
          this.h3usUI = new MiniMaxH3UnifiedSamplerUI(this);
        }
        return result;
      };

      nodeType.prototype.onSerialize = function () {
        const result = originalOnSerialize?.apply(this, arguments);
        snapshotWidgets(this);
        return result;
      };

      nodeType.prototype.onConfigure = function () {
        const result = originalOnConfigure?.apply(this, arguments);
        applyNodeVisuals(this);
        window.clearTimeout(this._h3usRestoreTimer);
        this._h3usRestoreTimer = window.setTimeout(() => {
          restoreWidgets(this);
          // Keep activeView in sync with restored workflow_mode — otherwise
          // F5 shows the old view's tab as white (view-active) while the
          // mode tab shows cyan (mode-active).
          try {
            const w = getWidget(this, "workflow_mode");
            const modeVal = w ? String(w.value) : null;
            const view = this.properties?.h3us_view;
            if (modeVal && WORKFLOW_MODES.includes(modeVal) && WORKFLOW_MODES.includes(view) && view !== modeVal) {
              this.properties.h3us_view = modeVal;
            }
          } catch {}
          if (!this.h3usUI) {
            this.h3usUI = new MiniMaxH3UnifiedSamplerUI(this);
          } else {
            this.h3usUI.rebuild();
          }
        }, 40);
        return result;
      };

      nodeType.prototype.onRemoved = function () {
        window.clearTimeout(this._h3usRestoreTimer);
        this._h3usRestoreTimer = null;
        this._h3usPovCleanup?.();
        this._h3usPovBuilt = false;
        this.h3usUI?.destroy();
        this.h3usUI = null;
        if (this._h3usOriginalSetSize) {
          this.setSize = this._h3usOriginalSetSize;
          this._h3usOriginalSetSize = null;
        }
        if (this._h3usOriginalComputeSize) {
          this.computeSize = this._h3usOriginalComputeSize;
          this._h3usOriginalComputeSize = null;
        }
        if (this._h3usOriginalOnDrawBackground) {
          this.onDrawBackground = this._h3usOriginalOnDrawBackground;
          this._h3usOriginalOnDrawBackground = null;
        }
        if (this._h3usOriginalOnDrawForeground) {
          this.onDrawForeground = this._h3usOriginalOnDrawForeground;
          this._h3usOriginalOnDrawForeground = null;
        }
        this._h3usCompactHeight = null;
        this._h3usVisualsPatched = false;
        return originalOnRemoved?.apply(this, arguments);
      };
      return;
    }

    // Config: progressive disclosure — splice to keep original suffix names.
    // Mirrors the native ReferenceToVideo Autogrow behavior but keeps the
    // "(REF2VA)" suffix and forceInput overrides. Splicing is the only
    // LiteGraph primitive that actually collapses node height.
    const CONFIG_SUFFIX = " (REF2VA)";
    const CONFIG_FAMILIES = [
      { prefix: "Ref Image ", type: "IMAGE", max: 9, gate: null },
      { prefix: "Ref Video ", type: "IMAGE", max: 3, gate: null },
      { prefix: "Ref Video Audio ", type: "AUDIO", max: 3, gate: "Ref Video " },
      { prefix: "Ref Audio ", type: "AUDIO", max: 3, gate: null },
    ];
    function cfgSlotInfo(name) {
      for (const f of CONFIG_FAMILIES) if (name?.startsWith(f.prefix) && name?.endsWith(CONFIG_SUFFIX)) {
        const n = parseInt(name.slice(f.prefix.length), 10);
        if (Number.isFinite(n)) return { fam: f, n };
      }
      return null;
    }
    function cfgHasLink(node, name) {
      const s = (node.inputs || []).find((i) => i && i.name === name);
      return Boolean(s && s.link != null);
    }
    function updateConfigSlots(node) {
      if (!node || !Array.isArray(node.inputs)) return;
      let changed = false;
      // Determine allowed max per family
      const allowedByFam = new Map();
      for (const fam of CONFIG_FAMILIES) {
        let hi = 0;
        for (const inp of node.inputs) {
          const info = cfgSlotInfo(inp?.name);
          if (info && info.fam === fam && inp.link != null && info.n > hi) hi = info.n;
        }
        let allowed = Math.max(1, hi);
        if (hi >= 1 && hi < fam.max) {
          const gateOk = !fam.gate || cfgHasLink(node, fam.gate + (hi + 1) + CONFIG_SUFFIX);
          if (!fam.gate || gateOk) allowed = hi + 1;
        }
        allowedByFam.set(fam, allowed);
      }
      // Remove tail beyond allowed
      for (let i = node.inputs.length - 1; i >= 0; i--) {
        const inp = node.inputs[i];
        const info = cfgSlotInfo(inp?.name);
        if (!info) continue;
        if (info.n > allowedByFam.get(info.fam) && inp.link == null) {
          node.removeInput(i);
          changed = true;
        }
      }
      // Add missing ordinals up to allowed (in canonical order)
      // Canonical order is the order declared in Python INPUT_TYPES:
      // First Frame, Last Frame, Ref Images 1..9, Ref Videos 1..3,
      // Ref Video Audios 1..3, Ref Audios 1..3, Frames/MegaPixels (override)
      const anchor = "Last Frame (I2V)";
      for (const fam of CONFIG_FAMILIES) {
        const allowed = allowedByFam.get(fam);
        for (let n = 1; n <= allowed; n++) {
          const name = fam.prefix + n + CONFIG_SUFFIX;
          if ((node.inputs || []).some((i) => i && i.name === name)) continue;
          // Find insertion index after anchor and earlier families
          let idx = node.inputs.findIndex((i) => i && i.name === anchor);
          // Walk forward to include earlier families' slots
          for (const earlier of CONFIG_FAMILIES) {
            if (earlier === fam) break;
            // find last slot of earlier family
            for (let j = node.inputs.length - 1; j > idx; j--) {
              if (node.inputs[j]?.name?.startsWith(earlier.prefix)) { idx = j; break; }
            }
          }
          // Also account for already-inserted lower ordinals of same family
          for (let k = 1; k < n; k++) {
            const prevName = fam.prefix + k + CONFIG_SUFFIX;
            const pos = node.inputs.findIndex((i) => i && i.name === prevName);
            if (pos !== -1 && pos > idx) idx = pos;
          }
          const insertAt = idx + 1;
          node.addInput(name, fam.type);
          // addInput appends at end — move it to insertAt
          const added = node.inputs.pop();
          node.inputs.splice(insertAt, 0, added);
          changed = true;
        }
      }
      if (changed) {
        try { node.setSize(node.computeSize()); } catch {}
        node.setDirtyCanvas?.(true, true);
      }
    }
    function scheduleConfigUpdate(node) {
      window.clearTimeout(node._h3usCfgTimer);
      node._h3usCfgTimer = window.setTimeout(() => updateConfigSlots(node), 20);
      requestAnimationFrame(() => updateConfigSlots(node));
    }
    nodeType.prototype.onNodeCreated = function () {
      const r = originalOnNodeCreated?.apply(this, arguments);
      scheduleConfigUpdate(this);
      return r;
    };
    nodeType.prototype.onConnectionsChange = function (...a) {
      const r = originalOnConnectionsChange?.apply(this, a);
      scheduleConfigUpdate(this);
      return r;
    };
    nodeType.prototype.onConfigure = function () {
      const r = originalOnConfigure?.apply(this, arguments);
      scheduleConfigUpdate(this);
      return r;
    };
    nodeType.prototype.onRemoved = function () {
      window.clearTimeout(this._h3usCfgTimer);
      return originalOnRemoved?.apply(this, arguments);
    };
  },
});

