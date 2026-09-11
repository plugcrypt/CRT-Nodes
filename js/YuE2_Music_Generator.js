import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

console.log("[YuE UI] Loading...");

const CSS_STYLES_YUE_UI = `
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --yue-primary: #8b5cf6;
    --yue-primary-light: #a78bfa;
    --yue-secondary: #22d3ee;
    --yue-accent: #f472b6;
    --yue-background: #08070f;
    --yue-surface: #17131f;
    --yue-border: #342c44;
    --yue-text: #ffffff;
    --yue-text-dim: #a29bb5;
    --yue-font: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    --yue-font-digital: 'Orbitron', 'Consolas', 'Courier New', monospace;
}

.yue-preview-container * {
    user-select: none !important;
    -webkit-user-select: none !important;
    scrollbar-width: thin;
    scrollbar-color: var(--yue-primary) var(--yue-surface);
}

.yue-preview-node-widget {
    position: relative !important;
    box-sizing: border-box !important;
    width: 100% !important;
    height: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: visible !important;
    display: block !important;
    visibility: visible !important;
    z-index: 999 !important;
    top: -10px !important;
}

.yue-preview-container {
    background: var(--yue-background);
    border: 2px solid var(--yue-primary);
    border-radius: 20px;
    padding: 16px;
    margin: 0;
    width: 100%;
    height: 100%;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(10px);
    gap: 12px;
    box-shadow:
        0 12px 40px rgba(139, 92, 246, 0.4),
        inset 0 2px 0 rgba(255, 255, 255, 0.1),
        0 0 60px rgba(34, 211, 238, 0.2);
}

.yue-preview-container::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(circle at 30% 20%, rgba(139, 92, 246, 0.12), transparent 50%),
                radial-gradient(circle at 70% 80%, rgba(34, 211, 238, 0.06), transparent 50%);
    pointer-events: none; z-index: 0;
}
.yue-preview-container > * { position: relative; z-index: 1; }

.yue-title {
    color: var(--yue-primary); font-family: var(--yue-font-digital); font-size: 18px; font-weight: 700;
    text-align: center; padding: 6px 0;
    text-shadow: 0 0 15px var(--yue-primary);
    background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.15), transparent);
    border-radius: 8px; margin-bottom: 4px;
}

.yue-led {
    width: 8px; height: 8px; border-radius: 50%;
    background: #444; position: absolute; top: 20px; right: 20px;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.8); z-index: 10;
}
.yue-led.ready { background: var(--yue-primary); box-shadow: 0 0 8px var(--yue-primary); }
.yue-led.generating { background: var(--yue-accent); animation: yueFastPulse 0.5s infinite; }
.yue-led.playing { background: var(--yue-secondary); }
@keyframes yueFastPulse { 0%,100% { opacity: 0.3; } 50% { opacity: 1; } }

.yue-tabs { display: flex; gap: 6px; border-bottom: 1px solid var(--yue-border); padding-bottom: 4px; }
.yue-tab-btn {
    background: rgba(0,0,0,0.3); border: 1px solid var(--yue-border);
    color: var(--yue-text-dim); font-family: var(--yue-font); font-size: 12px; font-weight: 500;
    padding: 8px; border-radius: 6px 6px 0 0; cursor: pointer; flex: 1; transition: all 0.2s;
}
.yue-tab-btn.active { background: rgba(139, 92, 246, 0.15); border-color: var(--yue-primary); color: var(--yue-primary); }

.yue-tab-content { display: none; flex-direction: column; gap: 10px; }
.yue-tab-content.active { display: flex; flex: 1; min-height: 0; }

.yue-section {
    background: rgba(0, 0, 0, 0.3); border: 1px solid var(--yue-border);
    border-radius: 12px; padding: 12px;
    display: flex; flex-direction: column; gap: 8px;
    flex: 1; min-height: 0; box-sizing: border-box; overflow-y: auto;
}
#tab-gen .yue-section { display: flex; flex-direction: column; }
#tab-settings .yue-section > .yue-sec { flex: 0 0 auto; }
#tab-settings .yue-sec-body > * { flex: 0 0 auto; }
#tab-gen .yue-textarea-container { flex: 1; min-height: 0; display: flex; flex-direction: column; }
.yue-textarea {
    flex: 1; min-height: 0; resize: none; background: rgba(0,0,0,0.4); border: 1px solid var(--yue-border);
    border-radius: 6px; padding: 8px; color: white; font-family: var(--yue-font); font-size: 12px; outline: none;
}
.yue-textarea:focus { border-color: var(--yue-primary); }
.yue-textarea:disabled { opacity: 0.5; border-style: dashed; font-style: italic; }

.yue-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.yue-input-group { display: flex; flex-direction: column; gap: 4px; }
.yue-input-group.full { grid-column: span 2; }
.yue-label { font-family: var(--yue-font); font-size: 11px; font-weight: 500; color: var(--yue-text-dim); text-transform: uppercase; letter-spacing: 0.03em; }
.yue-input, .yue-select {
    background: rgba(0,0,0,0.4); border: 1px solid var(--yue-border);
    border-radius: 6px; padding: 6px; color: white; font-family: var(--yue-font); font-size: 12px; outline: none;
}
.yue-select option {
    background: #000; color: white;
}

/* Sliders */
.yue-slider-row { display: flex; align-items: center; gap: 8px; }
.yue-slider { flex: 1; height: 4px; background: #333; -webkit-appearance: none; border-radius: 2px; }
.yue-slider::-webkit-slider-thumb { -webkit-appearance: none; width: 12px; height: 12px; background: var(--yue-primary); border-radius: 50%; cursor: pointer; }
.yue-val { font-family: var(--yue-font-digital); font-size: 11px; color: var(--yue-primary); width: 42px; text-align: right; }

.yue-sec { border: 1px solid var(--yue-border); border-radius: 10px; background: rgba(0,0,0,0.25); overflow: hidden; }
.yue-sec-head {
    display: flex; align-items: center; gap: 8px; width: 100%;
    padding: 9px 10px; background: transparent; border: none; cursor: pointer; text-align: left;
}
.yue-sec-head:hover { background: rgba(139, 92, 246, 0.08); }
.yue-sec-arrow { color: var(--yue-text-dim); font-size: 10px; width: 10px; flex: 0 0 10px; transition: transform 0.15s ease; }
.yue-sec.open .yue-sec-arrow { transform: rotate(90deg); color: var(--yue-primary); }
.yue-sec-title { font-family: var(--yue-font); font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--yue-text-dim); }
.yue-sec.open .yue-sec-title { color: var(--yue-text); }
.yue-sec-body { display: none; flex-direction: column; gap: 10px; padding: 2px 10px 10px 10px; }
.yue-sec.open .yue-sec-body { display: flex; }
.yue-gated { opacity: 0.35; pointer-events: none; }

.yue-gen-btn {
    background: linear-gradient(45deg, var(--yue-primary), var(--yue-primary-light));
    border: none; border-radius: 10px; padding: 12px; color: #000;
    font-family: var(--yue-font); font-weight: 600; font-size: 14px; cursor: pointer;
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3); margin-top: 5px;
    position: relative; overflow: hidden;
    display: flex; align-items: center; justify-content: center; line-height: 1;
}
.yue-gen-btn:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(139, 92, 246, 0.5); }
.yue-gen-btn:disabled { opacity: 1; cursor: not-allowed; }

.yue-gen-btn.generating { background: #222; box-shadow: inset 0 2px 5px rgba(0,0,0,0.5); }
.yue-gen-progress {
    position: absolute; top: 0; left: 0; bottom: 0; width: 0%;
    background: linear-gradient(90deg, rgba(139, 92, 246, 0.6), rgba(139, 92, 246, 0.8));
    transition: width 0.1s linear; z-index: 0;
}
.yue-gen-text { position: relative; z-index: 1; }
.yue-gen-btn.generating .yue-gen-text { color: white; }
.yue-gen-btn.success { background: linear-gradient(45deg, #22d3ee, #0ea5e9); box-shadow: 0 0 15px rgba(34, 211, 238, 0.4); }

.yue-wave-box {
    background: rgba(0,0,0,0.5); border: 2px solid var(--yue-border);
    border-radius: 12px; height: 120px; position: relative; overflow: hidden; cursor: pointer;
}
.yue-canvas { width: 100%; height: 100%; }
.yue-progress {
    position: absolute; top: 0; left: 0; height: 100%;
    background: linear-gradient(90deg, rgba(139,92,246,0.5), rgba(139,92,246,0.1));
    width: 0%; pointer-events: none;
}

.yue-controls {
    background: rgba(0,0,0,0.3); border: 1px solid var(--yue-border);
    border-radius: 12px; padding: 10px; display: flex; flex-direction: column; gap: 10px;
}
.yue-transport { display: flex; align-items: center; gap: 10px; }
.yue-transport-group { display: flex; align-items: center; gap: 6px; }
.yue-transport-group.pills { margin-left: 4px; }
.yue-btn-play {
    width: 42px; height: 42px; flex: 0 0 42px; border-radius: 50%; border: none;
    background: linear-gradient(135deg, var(--yue-secondary), #38bdf8); color: #04141a;
    font-size: 16px; cursor: pointer; display: flex; align-items: center; justify-content: center;
    box-shadow: 0 3px 10px rgba(34, 211, 238, 0.35); transition: transform 0.15s, box-shadow 0.15s;
}
.yue-btn-play:hover { transform: translateY(-1px); box-shadow: 0 5px 14px rgba(34, 211, 238, 0.5); }
.yue-btn-play.playing { background: linear-gradient(135deg, var(--yue-primary), var(--yue-primary-light)); color: #fff; box-shadow: 0 3px 10px rgba(139, 92, 246, 0.45); }
.yue-btn-icon {
    width: 34px; height: 34px; border-radius: 9px; border: 1px solid var(--yue-border);
    background: rgba(0,0,0,0.35); color: var(--yue-text-dim); cursor: pointer; font-size: 13px;
    display: flex; align-items: center; justify-content: center; transition: all 0.15s;
}
.yue-btn-icon:hover { color: #fff; border-color: #5b4f73; background: rgba(139, 92, 246, 0.12); }
.yue-btn-icon.active { color: var(--yue-primary); border-color: var(--yue-primary); background: rgba(139, 92, 246, 0.15); }

.yue-auto-switch {
    display: flex; align-items: center; gap: 6px; cursor: pointer;
    background: rgba(0,0,0,0.35); padding: 6px 10px; border-radius: 999px;
    border: 1px solid var(--yue-border); transition: all 0.15s;
}
.yue-auto-switch:hover { border-color: #5b4f73; }
.yue-auto-switch.active { border-color: var(--yue-primary); background: rgba(139, 92, 246, 0.12); }
.yue-auto-dot { width: 7px; height: 7px; border-radius: 50%; background: #555; transition: 0.2s; }
.yue-auto-switch.active .yue-auto-dot { background: var(--yue-primary); box-shadow: 0 0 6px var(--yue-primary); }
.yue-auto-label { font-family: var(--yue-font); font-size: 11px; font-weight: 500; color: var(--yue-text-dim); }
.yue-auto-switch.active .yue-auto-label { color: var(--yue-text); }
.yue-time { font-family: var(--yue-font-digital); font-size: 12px; color: var(--yue-secondary); background: rgba(0,0,0,0.4); padding: 6px 9px; border-radius: 7px; border: 1px solid var(--yue-border); white-space: nowrap; margin-left: auto; }

.yue-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
.yue-metric { background: rgba(0,0,0,0.4); border: 1px solid var(--yue-border); border-radius: 6px; padding: 4px; text-align: center; }
.yue-metric-label { font-size: 9px; color: var(--yue-text-dim); font-family: var(--yue-font); font-weight: 500; }
.yue-metric-val { font-size: 12px; color: var(--yue-primary); font-family: var(--yue-font-digital); font-weight: 700; }

.yue-check-row { display: flex; align-items: center; gap: 8px; }
.yue-checkbox { appearance: none; width: 14px; height: 14px; border: 1px solid var(--yue-primary); background: #222; cursor: pointer; position: relative; }
.yue-checkbox:checked::after { content: '✓'; position: absolute; top: -4px; left: 2px; color: var(--yue-primary); font-weight: bold; }
`;

class YuEPreviewUI {
    constructor(node) {
        this.node = node;
        this.isInitialized = false;
        this.isPlaying = false;
        this.isLooping = false;
        this.autoPreview = true;
        this.autoGenerate = true;
        this.isGenerating = false;
        this.currentTime = 0;
        this.duration = 0;
        this.volume = 0.8;
        this.node.title = "";
        this.node.bgcolor = "transparent";
        this.node.color = "transparent";

        this.node.setSize([650, 900]);

        try {
            this.initializeUI();
        } catch (e) {
            console.error("YuE UI Init Failed:", e);
        }
    }

    initializeUI() {
        if (this.isInitialized) return;
        this.injectStyles();
        this.hideOriginalWidgets();
        this.createCustomDOM();

        try {
            this.setupAudioContext();
        } catch (e) {
            console.warn("AudioContext init failed:", e);
        }

        this.setupEventHandlers();

        this.syncInterval = setInterval(() => this.syncInputsFromWidgets(), 500);

        this.onExecuting = ({ detail }) => {
            if (detail === this.node.id.toString()) {
                this.setGeneratingState();
            }
        };

        this.onProgress = ({ detail }) => {
            if (parseInt(detail.node) === this.node.id) {
                const pct = Math.round((detail.value / detail.max) * 100);
                if (this.dom && this.dom.genProgress) {
                    this.dom.genProgress.style.width = `${pct}%`;
                    this.dom.genText.textContent = `Generating... ${pct}%`;
                }
                if (this.dom && this.dom.genBtn && !this.dom.genBtn.classList.contains('generating')) {
                    this.setGeneratingState();
                }
            }
        };

        this.resetState = () => {
            this.isGenerating = false;
            this.setStatus('ready');
            if (this.dom && this.dom.genBtn) {
                this.dom.genBtn.disabled = false;
                this.dom.genBtn.classList.remove('generating');
                this.dom.genProgress.style.width = '0%';
                this.dom.genText.textContent = "Generate Song";
            }
        };

        api.addEventListener("executing", this.onExecuting);
        api.addEventListener("progress", this.onProgress);
        api.addEventListener("execution_error", this.resetState);
        api.addEventListener("execution_interrupted", this.resetState);

        this.isInitialized = true;
    }

    setGeneratingState() {
        this.isGenerating = true;
        this.setStatus('generating');
        if (this.dom && this.dom.genBtn) {
            this.dom.genBtn.disabled = true;
            this.dom.genBtn.classList.remove('success');
            this.dom.genBtn.classList.add('generating');
            this.dom.genText.textContent = "Initializing...";
            this.dom.genProgress.style.width = '0%';
        }
    }

    scheduleGenerate() {
        if (!this.autoGenerate || this.isGenerating) return;
        clearTimeout(this.genTimer);
        this.genTimer = setTimeout(() => {
            if (!this.autoGenerate || this.isGenerating) return;
            this.setGeneratingState();
            app.queuePrompt(0, 1);
        }, 1500);
    }

    injectStyles() {
        if (!document.getElementById('yue-preview-styles')) {
            const style = document.createElement('style');
            style.id = 'yue-preview-styles';
            style.textContent = CSS_STYLES_YUE_UI;
            document.head.appendChild(style);
        }
    }

    hideOriginalWidgets() {
        const WIDTH = 650;
        const HEIGHT = 900;
        if (this.node.widgets) {
            this.node.widgets.forEach(w => {
                if (w.type !== 'converted-widget') {
                    w.computeSize = () => [0, -4];
                    w.hidden = true;
                }
            });
        }
        // The custom UI stretches to fill the node (the lyrics area flexes),
        // so only minimums are enforced here; the node keeps whatever height
        // the user or the saved workflow gives it.
        if (!this.node._yueSizePatched) {
            this.node._yueSizePatched = true;
            this.node.computeSize = function (out) {
                const size = [Math.max(WIDTH, this.size?.[0] ?? WIDTH), Math.max(HEIGHT, this.size?.[1] ?? HEIGHT)];
                if (out) { out[0] = size[0]; out[1] = size[1]; return out; }
                return size;
            };
            // A height saved by an older version can be taller (blank body)
            // or shorter (clipped bottom) than the real content. Snap it to
            // the measured overlay height whenever the node is configured.
            const origConfigure = this.node.onConfigure;
            this.node.onConfigure = function () {
                const r = origConfigure?.apply(this, arguments);
                this.yueUI?.snapToContentSoon?.();
                return r;
            };
        }
        this.node.onResize = function (size) {
            if (this.flags?.collapsed) return;
            size[0] = Math.max(size[0], WIDTH);
            size[1] = Math.max(size[1], HEIGHT);
        };
        this.node.setSize([
            Math.max(WIDTH, this.node.size?.[0] ?? WIDTH),
            Math.max(HEIGHT, this.node.size?.[1] ?? HEIGHT),
        ]);
    }

    snapToContent() {
        const node = this.node;
        if (!node || node.flags?.collapsed) return;
        const wrap = this.widgetWrapper;
        if (!wrap || !wrap.isConnected) return;
        // scrollHeight is in layout units, i.e. canvas units like node.size
        // (unlike getBoundingClientRect, it is unaffected by canvas zoom).
        const contentH = wrap.scrollHeight || 0;
        if (!contentH || contentH < 100) return;
        const top = this.domWidget?.last_y ?? 100;
        const target = Math.round(top + contentH + 8);
        const w = Math.max(650, node.size?.[0] ?? 650);
        const h = node.size?.[1] ?? 0;
        if (Math.abs(h - target) > 4) node.setSize([w, Math.max(900, target)]);
    }

    snapToContentSoon() {
        requestAnimationFrame(() => this.snapToContent());
        if (document.fonts?.ready) {
            document.fonts.ready.then(() => this.snapToContent()).catch(() => {});
        }
    }

    createCustomDOM() {
        const widgetWrapper = document.createElement('div');
        widgetWrapper.className = 'yue-preview-node-widget';
        this.container = document.createElement('div');
        this.container.className = 'yue-preview-container';

        this.container.innerHTML = `
            <div class="yue-title">YuE2</div>
            <div class="yue-led"></div>
            <div class="yue-tabs">
                <button class="yue-tab-btn active" data-tab="gen">Generation</button>
                <button class="yue-tab-btn" data-tab="settings">Settings</button>
            </div>
            <div class="yue-tab-content active" id="tab-gen">
                <div class="yue-section">
                    <div class="yue-input-group full">
                        <label class="yue-label">Previous Runs</label>
                        <select class="yue-select" id="in-runs">
                            <option value="">-- select a run --</option>
                        </select>
                    </div>
                    <div class="yue-textarea-container">
                        <label class="yue-label">Lyrics</label>
                        <textarea class="yue-textarea" id="in-lyrics" placeholder="[Verse]..."></textarea>
                    </div>
                    <div class="yue-input-group">
                        <label class="yue-label">Style</label>
                        <textarea class="yue-textarea" id="in-style" style="min-height:60px; flex:none;" placeholder="English, warm piano pop, female voice, 88 BPM"></textarea>
                    </div>
                    <div class="yue-input-group full">
                        <label class="yue-label">Planning (CoT)</label>
                        <select class="yue-select" id="in-cot">
                            <option value="full">full - melody + chords</option>
                            <option value="melody">melody - melody only</option>
                            <option value="off">off - no plan</option>
                        </select>
                    </div>
                </div>
            </div>
            <div class="yue-tab-content" id="tab-settings">
                <div class="yue-section">
                    <div class="yue-sec open">
                        <button class="yue-sec-head" type="button"><span class="yue-sec-arrow">\u25B8</span><span class="yue-sec-title">Model &amp; Runtime</span></button>
                        <div class="yue-sec-body">
                            <div class="yue-grid">
                                <div class="yue-input-group">
                                    <label class="yue-label">Model</label>
                                    <select class="yue-select" id="in-model"></select>
                                </div>
                                <div class="yue-input-group">
                                    <label class="yue-label">Decoder (VAE)</label>
                                    <select class="yue-select" id="in-vae"></select>
                                </div>
                                <div class="yue-input-group">
                                    <label class="yue-label">Backend (AR)</label>
                                    <select class="yue-select" id="in-backend">
                                        <option value="torch">torch</option>
                                        <option value="torch-eager">torch-eager</option>
                                        <option value="vllm">vllm</option>
                                    </select>
                                </div>
                                <div class="yue-input-group">
                                    <label class="yue-label">Quantization</label>
                                    <select class="yue-select" id="in-quant">
                                        <option value="none">none</option>
                                        <option value="fp8">fp8</option>
                                    </select>
                                </div>
                                <div class="yue-input-group">
                                    <label class="yue-label">Attention (synthesis)</label>
                                    <select class="yue-select" id="in-attn">
                                        <option value="comfy">comfy</option>
                                        <option value="native">native</option>
                                        <option value="sage">sage</option>
                                        <option value="flash">flash</option>
                                    </select>
                                </div>
                                <div class="yue-input-group">
                                    <label class="yue-label">Model Before VAE Decode</label>
                                    <select class="yue-select" id="in-unload">
                                        <option value="disk">disk</option>
                                        <option value="cpu">cpu (RAM)</option>
                                        <option value="none">none</option>
                                    </select>
                                </div>
                                <div class="yue-input-group full">
                                    <label class="yue-label">Memory Budget (GiB)</label>
                                    <div class="yue-slider-row">
                                        <input type="range" class="yue-slider" id="in-mem" min="3" max="128" step="1">
                                        <span class="yue-val" id="val-mem">--</span>
                                    </div>
                                </div>
                            </div>
                            <label class="yue-check-row"><input type="checkbox" class="yue-checkbox" id="in-offload"><span class="yue-label">Offload AR During Synthesis</span></label>
                            <label class="yue-check-row"><input type="checkbox" class="yue-checkbox" id="in-auto" checked><span class="yue-label">Auto Download</span></label>
                            <label class="yue-check-row"><input type="checkbox" class="yue-checkbox" id="in-offload-run"><span class="yue-label">Offload Models After Run</span></label>
                        </div>
                    </div>

                    <div class="yue-sec">
                        <button class="yue-sec-head" type="button"><span class="yue-sec-arrow">\u25B8</span><span class="yue-sec-title">Sampling</span></button>
                        <div class="yue-sec-body">
                            <div class="yue-input-group full">
                                <label class="yue-label">Seed &amp; Control</label>
                                <div style="display:flex; gap:5px;">
                                    <input type="number" class="yue-input" id="in-seed" style="flex:1;">
                                    <select class="yue-select" id="in-seed-ctrl" style="width:100px;">
                                        <option value="fixed">Fixed</option>
                                        <option value="increment">Inc</option>
                                        <option value="decrement">Dec</option>
                                        <option value="randomize">Rand</option>
                                    </select>
                                </div>
                            </div>
                            <div class="yue-input-group full">
                                <label class="yue-label">CFG Scale (-1 = auto)</label>
                                <div class="yue-slider-row">
                                    <input type="range" class="yue-slider" id="in-cfg" min="-1" max="20" step="0.05">
                                    <span class="yue-val" id="val-cfg">--</span>
                                </div>
                            </div>
                            <label class="yue-check-row"><input type="checkbox" class="yue-checkbox" id="in-ovr"><span class="yue-label">Override Sampling</span></label>
                            <div id="sampling-gate" class="yue-gated" style="display:flex; flex-direction:column; gap:10px;">
                                <div class="yue-input-group full">
                                    <label class="yue-label">Temperature</label>
                                    <div class="yue-slider-row">
                                        <input type="range" class="yue-slider" id="in-temp" min="0" max="5" step="0.05">
                                        <span class="yue-val" id="val-temp">--</span>
                                    </div>
                                </div>
                                <div class="yue-input-group full">
                                    <label class="yue-label">Top P</label>
                                    <div class="yue-slider-row">
                                        <input type="range" class="yue-slider" id="in-topp" min="0.01" max="1" step="0.01">
                                        <span class="yue-val" id="val-topp">--</span>
                                    </div>
                                </div>
                                <div class="yue-input-group full">
                                    <label class="yue-label">Top K</label>
                                    <div class="yue-slider-row">
                                        <input type="range" class="yue-slider" id="in-topk" min="1" max="1000" step="1">
                                        <span class="yue-val" id="val-topk">--</span>
                                    </div>
                                </div>
                                <div class="yue-input-group full">
                                    <label class="yue-label">Repetition Penalty</label>
                                    <div class="yue-slider-row">
                                        <input type="range" class="yue-slider" id="in-reppen" min="0.1" max="3" step="0.01">
                                        <span class="yue-val" id="val-reppen">--</span>
                                    </div>
                                </div>
                                <div class="yue-input-group full">
                                    <label class="yue-label">Semantic Min Tokens</label>
                                    <div class="yue-slider-row">
                                        <input type="range" class="yue-slider" id="in-mintok" min="0" max="16000" step="50">
                                        <span class="yue-val" id="val-mintok">--</span>
                                    </div>
                                </div>
                                <div class="yue-input-group full">
                                    <label class="yue-label">Semantic Max Tokens</label>
                                    <div class="yue-slider-row">
                                        <input type="range" class="yue-slider" id="in-maxtok" min="200" max="16000" step="100">
                                        <span class="yue-val" id="val-maxtok">--</span>
                                    </div>
                                </div>
                                <div class="yue-input-group full">
                                    <label class="yue-label">ABC Max Tokens</label>
                                    <div class="yue-slider-row">
                                        <input type="range" class="yue-slider" id="in-abctok" min="32" max="8192" step="32">
                                        <span class="yue-val" id="val-abctok">--</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="yue-sec">
                        <button class="yue-sec-head" type="button"><span class="yue-sec-arrow">\u25B8</span><span class="yue-sec-title">Synthesis &amp; Decode</span></button>
                        <div class="yue-sec-body">
                            <div class="yue-input-group full">
                                <label class="yue-label">ODE Steps</label>
                                <div class="yue-slider-row">
                                    <input type="range" class="yue-slider" id="in-ode" min="1" max="64" step="1">
                                    <span class="yue-val" id="val-ode">--</span>
                                </div>
                            </div>
                            <div class="yue-input-group full">
                                <label class="yue-label">Max Length (s, 0 = model)</label>
                                <div class="yue-slider-row">
                                    <input type="range" class="yue-slider" id="in-maxlen" min="0" max="360" step="5">
                                    <span class="yue-val" id="val-maxlen">--</span>
                                </div>
                            </div>
                            <div class="yue-input-group full">
                                <label class="yue-label">VAE Core Frames (0 = auto)</label>
                                <div class="yue-slider-row">
                                    <input type="range" class="yue-slider" id="in-vaeframes" min="0" max="8192" step="128">
                                    <span class="yue-val" id="val-vaeframes">--</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="yue-sec">
                        <button class="yue-sec-head" type="button"><span class="yue-sec-arrow">\u25B8</span><span class="yue-sec-title">Output</span></button>
                        <div class="yue-sec-body">
                            <label class="yue-check-row"><input type="checkbox" class="yue-checkbox" id="in-save" checked><span class="yue-label">Save Output to Disk</span></label>
                            <input type="text" class="yue-input" id="in-path" placeholder="./ComfyUI/output/Music_YuE2" style="width:100%">
                        </div>
                    </div>
                </div>
            </div>
            <button class="yue-gen-btn" id="btn-gen">
                <div class="yue-gen-progress"></div>
                <span class="yue-gen-text">Generate Song</span>
            </button>
            <div class="yue-wave-box">
                <div class="yue-progress"></div>
                <canvas class="yue-canvas"></canvas>
            </div>
            <div class="yue-controls">
                <div class="yue-transport">
                    <div class="yue-transport-group">
                        <button class="yue-btn-play" type="button" title="Play / Pause">\u25B6</button>
                        <button class="yue-btn-icon" id="btn-stop" type="button" title="Stop">\u25A0</button>
                        <button class="yue-btn-icon" id="btn-loop" type="button" title="Loop">\u27F2</button>
                    </div>
                    <div class="yue-transport-group pills">
                        <div class="yue-auto-switch active" id="btn-auto-preview" title="Auto-play after generation">
                            <div class="yue-auto-dot"></div>
                            <span class="yue-auto-label">Auto</span>
                        </div>
                        <div class="yue-auto-switch active" id="btn-auto-gen" title="Auto-queue after control changes">
                            <div class="yue-auto-dot"></div>
                            <span class="yue-auto-label">AutoGen</span>
                        </div>
                    </div>
                    <div class="yue-time">00:00 / 00:00</div>
                </div>
                <div class="yue-metrics">
                    <div class="yue-metric"><div class="yue-metric-label">PEAK</div><div class="yue-metric-val" data-id="peak">--</div></div>
                    <div class="yue-metric"><div class="yue-metric-label">RMS</div><div class="yue-metric-val" data-id="rms">--</div></div>
                    <div class="yue-metric"><div class="yue-metric-label">LUFS</div><div class="yue-metric-val" data-id="lufs">--</div></div>
                </div>
                <div class="yue-slider-row">
                    <span class="yue-label" style="width:30px;">VOL</span>
                    <input type="range" class="yue-slider" id="in-vol" min="0" max="2" step="0.01" value="0.8">
                    <span class="yue-val" id="val-vol">80%</span>
                </div>
            </div>`;

        widgetWrapper.appendChild(this.container);
        this.widgetWrapper = widgetWrapper;
        this.domWidget = this.node.addDOMWidget('yue_preview_ui', 'div', widgetWrapper, { serialize: false });

        const q = (sel) => this.container.querySelector(sel);
        this.dom = {
            tabs: this.container.querySelectorAll('.yue-tab-btn'),
            contents: this.container.querySelectorAll('.yue-tab-content'),
            lyrics: q('#in-lyrics'), style: q('#in-style'), model: q('#in-model'), vae: q('#in-vae'), cot: q('#in-cot'),
            runs: q('#in-runs'),
            cfg: q('#in-cfg'), valCfg: q('#val-cfg'),
            ode: q('#in-ode'), valOde: q('#val-ode'),
            maxlen: q('#in-maxlen'), valMaxlen: q('#val-maxlen'),
            vaeframes: q('#in-vaeframes'), valVaeframes: q('#val-vaeframes'),
            attn: q('#in-attn'),
            seed: q('#in-seed'), seedCtrl: q('#in-seed-ctrl'),
            backend: q('#in-backend'), quant: q('#in-quant'),
            unload: q('#in-unload'),
            mem: q('#in-mem'), valMem: q('#val-mem'),
            ovr: q('#in-ovr'), samplingGate: q('#sampling-gate'),
            temp: q('#in-temp'), valTemp: q('#val-temp'),
            topp: q('#in-topp'), valTopp: q('#val-topp'),
            topk: q('#in-topk'), valTopk: q('#val-topk'),
            reppen: q('#in-reppen'), valReppen: q('#val-reppen'),
            mintok: q('#in-mintok'), valMintok: q('#val-mintok'),
            maxtok: q('#in-maxtok'), valMaxtok: q('#val-maxtok'),
            abctok: q('#in-abctok'), valAbctok: q('#val-abctok'),
            sections: this.container.querySelectorAll('.yue-sec'),
            offload: q('#in-offload'), offloadRun: q('#in-offload-run'), auto: q('#in-auto'),
            save: q('#in-save'), path: q('#in-path'),
            genBtn: q('#btn-gen'), genProgress: q('.yue-gen-progress'), genText: q('.yue-gen-text'),
            waveBox: q('.yue-wave-box'), canvas: q('.yue-canvas'), progress: q('.yue-progress'),
            playBtn: q('.yue-btn-play'), stopBtn: q('#btn-stop'), loopBtn: q('#btn-loop'), autoSwitch: q('#btn-auto-preview'),
            autoGenSwitch: q('#btn-auto-gen'),
            time: q('.yue-time'), vol: q('#in-vol'), valVol: q('#val-vol'),
            peak: q('[data-id="peak"]'), rms: q('[data-id="rms"]'), lufs: q('[data-id="lufs"]'),
            led: q('.yue-led')
        };
        this.waveformCtx = this.dom.canvas.getContext('2d');
        this.setStatus('ready');
        this.populateModelOptions();
    }

    setupEventHandlers() {
        this.dom.tabs.forEach(btn => {
            btn.addEventListener('click', () => {
                this.dom.tabs.forEach(b => b.classList.remove('active'));
                this.dom.contents.forEach(c => c.classList.remove('active'));
                btn.classList.add('active');
                this.container.querySelector(`#tab-${btn.dataset.tab}`).classList.add('active');
                if (btn.dataset.tab === 'gen') this.refreshRuns();
            });
        });

        this.dom.sections.forEach(sec => {
            const head = sec.querySelector('.yue-sec-head');
            if (!head) return;
            head.addEventListener('click', () => {
                sec.classList.toggle('open');
                this.updateSamplingGate();
            });
        });

        this.dom.playBtn.addEventListener('click', () => this.handlePlayButton());
        this.dom.stopBtn.addEventListener('click', () => this.stopPlayback());

        this.dom.loopBtn.addEventListener('click', () => {
            this.isLooping = !this.isLooping;
            this.dom.loopBtn.classList.toggle('active', this.isLooping);
        });

        this.dom.autoSwitch.addEventListener('click', () => {
            this.autoPreview = !this.autoPreview;
            this.dom.autoSwitch.classList.toggle('active', this.autoPreview);
        });

        this.dom.autoGenSwitch.addEventListener('click', () => {
            this.autoGenerate = !this.autoGenerate;
            this.dom.autoGenSwitch.classList.toggle('active', this.autoGenerate);
            if (this.autoGenerate) this.scheduleGenerate();
        });

        this.dom.vol.addEventListener('input', (e) => {
            this.volume = parseFloat(e.target.value);
            this.dom.valVol.textContent = Math.round(this.volume * 100) + "%";
            if (this.gainNode) this.gainNode.gain.value = this.volume;
        });

        this.dom.genBtn.addEventListener('click', () => {
            this.setGeneratingState();
            app.queuePrompt(0, 1);
        });

        this.dom.runs.addEventListener('mousedown', () => this.refreshRuns());
        this.dom.runs.addEventListener('focus', () => this.refreshRuns());
        this.dom.runs.addEventListener('change', (e) => {
            if (e.target.value) this.loadRun(e.target.value);
        });

        this.dom.waveBox.addEventListener('mousedown', (e) => {
            this.isDraggingProgress = true;
            this.seekToPosition(e);
        });
        document.addEventListener('mousemove', (e) => { if (this.isDraggingProgress) this.seekToPosition(e); });
        document.addEventListener('mouseup', () => this.isDraggingProgress = false);

        this.bindInput('lyrics', this.dom.lyrics);
        this.bindInput('style', this.dom.style);
        this.bindInput('model', this.dom.model, 'change');
        this.bindInput('vae', this.dom.vae, 'change');
        this.bindInput('cot', this.dom.cot, 'change');
        this.bindSlider('cfg_scale', this.dom.cfg, this.dom.valCfg);
        this.bindSlider('ode_steps', this.dom.ode, this.dom.valOde);
        this.bindSlider('max_length_seconds', this.dom.maxlen, this.dom.valMaxlen);
        this.bindSlider('vae_core_frames', this.dom.vaeframes, this.dom.valVaeframes);
        this.bindSlider('memory_budget_gib', this.dom.mem, this.dom.valMem);
        this.bindInput('attention', this.dom.attn, 'change');
        this.bindInput('backend', this.dom.backend, 'change');
        this.bindInput('quantization', this.dom.quant, 'change');
        this.bindInput('unload_model_before_vae', this.dom.unload, 'change');
        this.bindSlider('temperature', this.dom.temp, this.dom.valTemp);
        this.bindSlider('top_p', this.dom.topp, this.dom.valTopp);
        this.bindSlider('top_k', this.dom.topk, this.dom.valTopk);
        this.bindSlider('repetition_penalty', this.dom.reppen, this.dom.valReppen);
        this.bindSlider('semantic_min_tokens', this.dom.mintok, this.dom.valMintok);
        this.bindSlider('semantic_max_tokens', this.dom.maxtok, this.dom.valMaxtok);
        this.bindSlider('abc_max_tokens', this.dom.abctok, this.dom.valAbctok);
        this.bindCheckbox('override_sampling', this.dom.ovr);
        this.dom.ovr.addEventListener('change', () => this.updateSamplingGate());
        this.bindCheckbox('offload_ar', this.dom.offload);
        this.bindCheckbox('auto_download', this.dom.auto);
        this.bindCheckbox('offload_after_run', this.dom.offloadRun);
        this.bindCheckbox('save_output', this.dom.save, false);
        this.bindInput('save_path', this.dom.path, 'input', false);

        ['model', 'vae', 'cot', 'attn', 'backend', 'quant', 'unload', 'seed-ctrl'].forEach(id => {
            const el = this.container.querySelector(`#in-${id}`);
            if (el) el.addEventListener('mousedown', (e) => e.stopPropagation());
        });

        this.dom.seed.addEventListener('input', (e) => {
            const w = this.node.widgets.find(w => w.name === 'seed');
            if (w) w.value = Number(e.target.value);
            this.scheduleGenerate();
        });
        this.dom.seedCtrl.addEventListener('change', (e) => {
            const w = this.node.widgets.find(w => w.name === 'control_after_generate');
            if (w) w.value = e.target.value;
        });

        this.updateSamplingGate();
        this.refreshRuns();
        this.snapToContentSoon();
    }

    getSavePath() {
        const w = this.node.widgets?.find(x => x.name === 'save_path');
        return w?.value || './ComfyUI/output/Music_YuE2';
    }

    async refreshRuns() {
        const path = this.getSavePath();
        try {
            const res = await fetch(`/crt/yue2/runs?path=${encodeURIComponent(path)}`);
            if (!res.ok) return;
            const data = await res.json();
            const runs = data.runs || [];
            const current = this.dom.runs.value;
            this.dom.runs.innerHTML = '<option value="">-- select a run --</option>';
            for (const name of runs) {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                this.dom.runs.appendChild(opt);
            }
            if (runs.includes(current)) this.dom.runs.value = current;
        } catch (e) {
            console.error("YuE: failed to list previous runs", e);
        }
    }

    async loadRun(name) {
        const path = this.getSavePath();
        try {
            const res = await fetch(`/crt/yue2/run?path=${encodeURIComponent(path)}&name=${encodeURIComponent(name)}`);
            if (!res.ok) return;
            const data = await res.json();
            this.applyPreset(data.settings || data);
        } catch (e) {
            console.error("YuE: failed to load run preset", e);
        }
    }

    applyPreset(settings) {
        if (!settings) return;
        const setWidget = (name, value) => {
            const w = this.node.widgets?.find(x => x.name === name);
            if (w && value !== undefined && value !== null) w.value = value;
        };
        const setInput = (name, dom, value) => {
            setWidget(name, value);
            if (dom && value !== undefined && value !== null) dom.value = value;
        };
        const setSlider = (name, dom, disp, value) => {
            setWidget(name, value);
            if (dom && value !== undefined && value !== null) {
                dom.value = value;
                if (disp) disp.textContent = value;
            }
        };
        const setCheck = (name, dom, value) => {
            setWidget(name, value);
            if (dom && value !== undefined && value !== null) dom.checked = Boolean(value);
        };

        setInput('model', this.dom.model, settings.model);
        setInput('vae', this.dom.vae, settings.vae);
        setInput('style', this.dom.style, settings.style);
        setInput('lyrics', this.dom.lyrics, settings.lyrics);
        setInput('cot', this.dom.cot, settings.cot);
        setInput('seed', this.dom.seed, settings.seed);
        setWidget('abc', settings.abc);
        setSlider('cfg_scale', this.dom.cfg, this.dom.valCfg, settings.cfg_scale);
        setSlider('ode_steps', this.dom.ode, this.dom.valOde, settings.ode_steps);
        setSlider('max_length_seconds', this.dom.maxlen, this.dom.valMaxlen, settings.max_length_seconds);
        setSlider('vae_core_frames', this.dom.vaeframes, this.dom.valVaeframes, settings.vae_core_frames);
        setSlider('memory_budget_gib', this.dom.mem, this.dom.valMem, settings.memory_budget_gib);
        setInput('attention', this.dom.attn, settings.attention);
        setInput('backend', this.dom.backend, settings.backend);
        setInput('quantization', this.dom.quant, settings.quantization);
        setInput('unload_model_before_vae', this.dom.unload, settings.unload_model_before_vae);
        setSlider('temperature', this.dom.temp, this.dom.valTemp, settings.temperature);
        setSlider('top_p', this.dom.topp, this.dom.valTopp, settings.top_p);
        setSlider('top_k', this.dom.topk, this.dom.valTopk, settings.top_k);
        setSlider('repetition_penalty', this.dom.reppen, this.dom.valReppen, settings.repetition_penalty);
        setSlider('semantic_min_tokens', this.dom.mintok, this.dom.valMintok, settings.semantic_min_tokens);
        setSlider('semantic_max_tokens', this.dom.maxtok, this.dom.valMaxtok, settings.semantic_max_tokens);
        setSlider('abc_max_tokens', this.dom.abctok, this.dom.valAbctok, settings.abc_max_tokens);
        setCheck('override_sampling', this.dom.ovr, settings.override_sampling);
        setCheck('offload_ar', this.dom.offload, settings.offload_ar);
        setCheck('auto_download', this.dom.auto, settings.auto_download);
        setCheck('offload_after_run', this.dom.offloadRun, settings.offload_after_run);
        setCheck('save_output', this.dom.save, settings.save_output);
        setInput('save_path', this.dom.path, settings.save_path);

        this.updateSamplingGate();
    }

    populateModelOptions() {
        const fill = (widgetName, dom) => {
            const widget = this.node.widgets?.find(w => w.name === widgetName);
            if (widget?.options?.values) {
                dom.innerHTML = '';
                widget.options.values.forEach(v => {
                    const opt = document.createElement('option');
                    opt.value = v; opt.textContent = v;
                    dom.appendChild(opt);
                });
                if (widget.value !== undefined) dom.value = widget.value;
            }
        };
        fill('model', this.dom.model);
        fill('vae', this.dom.vae);
    }

    bindInput(wName, dom, evt = 'input', trigger = true) {
        dom.addEventListener(evt, (e) => {
            const w = this.node.widgets.find(x => x.name === wName);
            if (w) w.value = e.target.value;
            if (trigger) this.scheduleGenerate();
        });
    }
    bindSlider(wName, dom, valDom, trigger = true) {
        dom.addEventListener('input', (e) => {
            valDom.textContent = e.target.value;
            const w = this.node.widgets.find(x => x.name === wName);
            if (w) w.value = (dom.step % 1 !== 0) ? parseFloat(e.target.value) : parseInt(e.target.value);
            if (trigger) this.scheduleGenerate();
        });
    }
    bindCheckbox(wName, dom, trigger = true) {
        const w = this.node.widgets.find(x => x.name === wName);
        if (w) {
            if (wName === 'save_output') w.value = true;
            dom.checked = w.value;
        }
        dom.addEventListener('change', (e) => {
            if (w) w.value = e.target.checked;
            if (trigger) this.scheduleGenerate();
        });
    }

    updateSamplingGate() {
        const w = this.node.widgets.find(x => x.name === 'override_sampling');
        const enabled = w ? Boolean(w.value) : false;
        if (this.dom.samplingGate) this.dom.samplingGate.classList.toggle('yue-gated', !enabled);
    }

    syncInputsFromWidgets() {
        if (!this.node) return;

        const checkOverride = (overrideName, dom) => {
            const idx = this.node.inputs?.findIndex(i => i.name === overrideName && i.link !== null);
            if (idx > -1) {
                dom.disabled = true;
                return true;
            }
            dom.disabled = false;
            return false;
        };

        if (!checkOverride('lyrics_override', this.dom.lyrics)) {
            const w = this.node.widgets.find(x => x.name === 'lyrics');
            if (w && document.activeElement !== this.dom.lyrics) this.dom.lyrics.value = w.value;
        }
        if (!checkOverride('style_override', this.dom.style)) {
            const w = this.node.widgets.find(x => x.name === 'style');
            if (w && document.activeElement !== this.dom.style) this.dom.style.value = w.value;
        }

        const syncS = (name, dom, disp) => {
            const w = this.node.widgets.find(x => x.name === name);
            if (w && dom.value != w.value) {
                dom.value = w.value;
                disp.textContent = w.value;
            }
        };
        syncS('cfg_scale', this.dom.cfg, this.dom.valCfg);
        syncS('ode_steps', this.dom.ode, this.dom.valOde);
        syncS('max_length_seconds', this.dom.maxlen, this.dom.valMaxlen);
        syncS('vae_core_frames', this.dom.vaeframes, this.dom.valVaeframes);
        syncS('memory_budget_gib', this.dom.mem, this.dom.valMem);
        syncS('temperature', this.dom.temp, this.dom.valTemp);
        syncS('top_p', this.dom.topp, this.dom.valTopp);
        syncS('top_k', this.dom.topk, this.dom.valTopk);
        syncS('repetition_penalty', this.dom.reppen, this.dom.valReppen);
        syncS('semantic_min_tokens', this.dom.mintok, this.dom.valMintok);
        syncS('semantic_max_tokens', this.dom.maxtok, this.dom.valMaxtok);
        syncS('abc_max_tokens', this.dom.abctok, this.dom.valAbctok);

        const syncSel = (name, dom) => {
            const w = this.node.widgets.find(x => x.name === name);
            if (w && dom.value !== w.value) dom.value = w.value;
        };
        syncSel('cot', this.dom.cot);
        syncSel('attention', this.dom.attn);
        syncSel('backend', this.dom.backend);
        syncSel('quantization', this.dom.quant);
        syncSel('unload_model_before_vae', this.dom.unload);

        if (!checkOverride('seed_override', this.dom.seed)) {
            const wS = this.node.widgets.find(x => x.name === 'seed');
            if (wS && this.dom.seed.value != wS.value) this.dom.seed.value = wS.value;
            this.dom.seedCtrl.disabled = false;
        } else {
            this.dom.seedCtrl.disabled = true;
        }

        const wC = this.node.widgets.find(x => x.name === 'control_after_generate');
        if (wC && this.dom.seedCtrl.value !== wC.value) {
            this.dom.seedCtrl.value = wC.value;
        }

        const wSave = this.node.widgets.find(x => x.name === 'save_output');
        if (wSave) this.dom.save.checked = wSave.value;

        const wPath = this.node.widgets.find(x => x.name === 'save_path');
        if (wPath && document.activeElement !== this.dom.path) this.dom.path.value = wPath.value;

        const wOvr = this.node.widgets.find(x => x.name === 'override_sampling');
        if (wOvr) this.dom.ovr.checked = Boolean(wOvr.value);
        this.updateSamplingGate();
    }

    setupAudioContext() {
        if (!window.yueAudioContext) window.yueAudioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.audioContext = window.yueAudioContext;
        this.gainNode = this.audioContext.createGain();
        this.gainNode.connect(this.audioContext.destination);

        // Parallel, inaudible K-weighting branch (ITU-R BS.1770) feeding a
        // processor that accumulates the played samples for a live LUFS read.
        this.lufsHighShelf = this.audioContext.createBiquadFilter();
        this.lufsHighShelf.type = 'highshelf';
        this.lufsHighShelf.frequency.value = 1681.974450955533;
        this.lufsHighShelf.gain.value = 3.999843853973347;
        this.lufsHighShelf.Q.value = 0.7071752369554196;

        this.lufsHighPass = this.audioContext.createBiquadFilter();
        this.lufsHighPass.type = 'highpass';
        this.lufsHighPass.frequency.value = 38.13547087602444;
        this.lufsHighPass.Q.value = 0.5003270373238773;

        this.lufsProcessor = this.audioContext.createScriptProcessor(4096, 2, 2);
        this.lufsSilence = this.audioContext.createGain();
        this.lufsSilence.gain.value = 0;

        this.lufsHighShelf.connect(this.lufsHighPass);
        this.lufsHighPass.connect(this.lufsProcessor);
        this.lufsProcessor.connect(this.lufsSilence);
        this.lufsSilence.connect(this.audioContext.destination);

        this.lufsProcessor.onaudioprocess = (e) => {
            if (!this.isPlaying) return;
            const inBuf = e.inputBuffer;
            const n = inBuf.length;
            let sumSq = 0;
            for (let c = 0; c < inBuf.numberOfChannels; c++) {
                const data = inBuf.getChannelData(c);
                for (let i = 0; i < n; i++) sumSq += data[i] * data[i];
            }
            this.lufsSumSq += sumSq;
            this.lufsSamples += n;
            const meanSq = this.lufsSumSq / this.lufsSamples;
            if (meanSq > 0) this.lufsValue = -0.691 + 10 * Math.log10(meanSq);
        };

        this.resetLufs();
    }

    resetLufs() {
        this.lufsSumSq = 0;
        this.lufsSamples = 0;
        this.lufsValue = null;
        if (this.dom && this.dom.lufs) this.dom.lufs.textContent = "--";
    }

    handlePlayButton() {
        if (!this.generatedAudioBuffer) return;
        this.isPlaying ? this.pausePlayback() : this.startPlayback();
    }

    startPlayback() {
        if (!this.generatedAudioBuffer) return;
        if (this.sourceNode) this.sourceNode.stop();
        if (this.audioContext.state === 'suspended') this.audioContext.resume();

        // A fresh start (or loop/seek to the top) restarts LUFS monitoring;
        // resuming from a paused position keeps accumulating.
        if (this.currentTime <= 0.01) this.resetLufs();

        this.sourceNode = this.audioContext.createBufferSource();
        this.sourceNode.buffer = this.generatedAudioBuffer;
        this.sourceNode.connect(this.gainNode);
        if (this.lufsHighShelf) this.sourceNode.connect(this.lufsHighShelf);
        this.gainNode.gain.value = this.volume;

        this.sourceNode.start(0, this.currentTime);
        this.startTime = this.audioContext.currentTime - this.currentTime;
        this.isPlaying = true;

        this.dom.playBtn.innerHTML = '⏸';
        this.dom.playBtn.classList.add('playing');
        this.setStatus('playing');

        this.sourceNode.onended = () => {
            if (this.isPlaying) {
                if (this.isLooping) {
                    this.currentTime = 0;
                    this.startPlayback();
                } else if (this.currentTime >= this.duration - 0.1) {
                    this.stopPlayback();
                }
            }
        };
        this.updateVis();
    }

    pausePlayback() {
        if (this.sourceNode) {
            try { this.sourceNode.stop(); } catch (e) {}
            this.currentTime = this.audioContext.currentTime - this.startTime;
        }
        this.isPlaying = false;
        this.dom.playBtn.innerHTML = '▶';
        this.dom.playBtn.classList.remove('playing');
        this.setStatus('ready');
        cancelAnimationFrame(this.raf);
    }

    stopPlayback() {
        if (this.sourceNode) {
            try { this.sourceNode.stop(); } catch (e) {}
            this.sourceNode.disconnect();
            this.sourceNode = null;
        }
        this.isPlaying = false;
        this.currentTime = 0;
        this.dom.playBtn.innerHTML = '▶';
        this.dom.playBtn.classList.remove('playing');
        this.setStatus('ready');
        this.resetLufs();
        this.updateProgress();
        cancelAnimationFrame(this.raf);
    }

    seekToPosition(e) {
        if (!this.generatedAudioBuffer) return;
        const rect = this.dom.waveBox.getBoundingClientRect();
        const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        this.currentTime = pct * this.duration;
        this.updateProgress();
        if (this.isPlaying) this.startPlayback();
    }

    updateVis() {
        if (!this.isPlaying) return;
        this.currentTime = this.audioContext.currentTime - this.startTime;
        this.updateProgress();
        this.dom.lufs.textContent = this.lufsValue != null ? this.lufsValue.toFixed(1) + " LUFS" : "--";
        this.raf = requestAnimationFrame(() => this.updateVis());
    }

    updateProgress() {
        const fmt = (s) => {
            const m = Math.floor(s / 60);
            const sc = Math.floor(s % 60);
            return `${m}:${sc.toString().padStart(2, '0')}`;
        };
        this.dom.time.textContent = `${fmt(this.currentTime)} / ${fmt(this.duration)}`;
        if (this.duration > 0) {
            this.dom.progress.style.width = ((this.currentTime / this.duration) * 100) + '%';
        }
    }

    setStatus(s) {
        if (this.dom && this.dom.led) {
            this.dom.led.className = 'yue-led ' + s;
        }
    }

    async loadGeneratedAudio(audioInfo, metrics) {
        try {
            this.isGenerating = false;
            this.setStatus('ready');
            this.dom.genBtn.disabled = false;
            this.dom.genBtn.classList.remove('generating');
            this.dom.genBtn.classList.add('success');
            this.dom.genText.textContent = "Generate Again";
            this.dom.genProgress.style.width = '0%';

            if (audioInfo?.filename) {
                const p = new URLSearchParams({ filename: audioInfo.filename, subfolder: audioInfo.subfolder, type: audioInfo.type });
                const res = await fetch('/view?' + p);
                const buf = await res.arrayBuffer();

                if (this.audioContext.state === 'suspended') await this.audioContext.resume();
                this.generatedAudioBuffer = await this.audioContext.decodeAudioData(buf);
                this.duration = this.generatedAudioBuffer.duration;
                this.currentTime = 0;

                this.drawWave();
                this.updateProgress();
                this.resetLufs();

                if (metrics) {
                    this.dom.peak.textContent = metrics.peak ? metrics.peak + " dB" : "--";
                    this.dom.rms.textContent = metrics.rms ? metrics.rms + " dB" : "--";
                }

                if (this.autoPreview) {
                    setTimeout(() => this.startPlayback(), 100);
                }
            }
            this.refreshRuns();
        } catch (e) {
            console.error(e);
            this.setStatus('error');
            this.resetState();
        }
    }

    drawWave() {
        if (!this.generatedAudioBuffer) return;
        const w = this.dom.canvas.width = this.dom.waveBox.clientWidth;
        const h = this.dom.canvas.height = this.dom.waveBox.clientHeight;
        const ctx = this.waveformCtx;
        const data = this.generatedAudioBuffer.getChannelData(0);
        const step = Math.ceil(data.length / w);
        const amp = h / 2;

        ctx.clearRect(0, 0, w, h);
        ctx.strokeStyle = '#8b5cf6';
        ctx.beginPath();
        for (let i = 0; i < w; i++) {
            let min = 1.0, max = -1.0;
            for (let j = 0; j < step; j++) {
                const d = data[(i * step) + j];
                if (d < min) min = d;
                if (d > max) max = d;
            }
            ctx.moveTo(i, (1 + min) * amp);
            ctx.lineTo(i, (1 + max) * amp);
        }
        ctx.stroke();
    }

    destroy() {
        this.stopPlayback();
        clearInterval(this.syncInterval);
        clearTimeout(this.genTimer);
        if (this.lufsProcessor) {
            this.lufsProcessor.onaudioprocess = null;
            try { this.lufsProcessor.disconnect(); } catch (e) {}
            this.lufsProcessor = null;
        }
        for (const node of [this.lufsHighShelf, this.lufsHighPass, this.lufsSilence]) {
            if (node) { try { node.disconnect(); } catch (e) {} }
        }
        this.lufsHighShelf = this.lufsHighPass = this.lufsSilence = null;
        if (api.removeEventListener) {
            api.removeEventListener("executing", this.onExecuting);
            api.removeEventListener("progress", this.onProgress);
            api.removeEventListener("execution_error", this.resetState);
            api.removeEventListener("execution_interrupted", this.resetState);
        }
        if (this.domWidget) {
            const idx = this.node.widgets?.indexOf(this.domWidget);
            if (idx > -1) this.node.widgets.splice(idx, 1);
            this.domWidget.onRemove?.();
            this.domWidget = null;
        }
        if (this.widgetWrapper) { this.widgetWrapper.remove(); this.widgetWrapper = null; }
        if (this.container) this.container.remove();
    }
}

app.registerExtension({
    name: "Comfy.YuE.UI",
    async nodeCreated(node) {
        if (node.comfyClass !== "YuEMusicGenerator") return;
        if (node.yueUI) node.yueUI.destroy();
        const ui = new YuEPreviewUI(node);
        node.yueUI = ui;

        const applyOutput = (ui_data) => {
            if (!ui_data) return;
            if (ui.dom) {
                if (ui_data?.style?.[0] !== undefined) ui.dom.style.value = ui_data.style[0];
                if (ui_data?.lyrics?.[0] !== undefined) ui.dom.lyrics.value = ui_data.lyrics[0];
            }
            if (ui_data?.audio?.[0]) ui.loadGeneratedAudio(ui_data.audio[0], ui_data.metrics?.[0]);
        };

        // node.onExecuted only fires for the workflow that is currently on
        // screen, so a run that finishes while another workflow tab is open
        // never reaches this node. Listen to the raw event and replay the
        // output on the next draw, i.e. once this node is visible again.
        let pendingOutput = null;
        const onExecuted = ({ detail }) => {
            if (!detail || String(detail.node) !== String(node.id)) return;
            pendingOutput = detail.output;
            node.setDirtyCanvas?.(true, false);
        };
        api.addEventListener("executed", onExecuted);

        const origDraw = node.onDrawForeground;
        node.onDrawForeground = function () {
            if (pendingOutput) {
                const out = pendingOutput;
                pendingOutput = null;
                applyOutput(out);
            }
            return origDraw?.apply(this, arguments);
        };

        const onRemoved = node.onRemoved;
        node.onRemoved = function () {
            if (api.removeEventListener) api.removeEventListener("executed", onExecuted);
            return onRemoved?.apply(this, arguments);
        };
    }
});
