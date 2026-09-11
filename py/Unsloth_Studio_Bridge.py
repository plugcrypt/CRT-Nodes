import os
import re
import json
import sys
import base64
import io
import time
import hmac
import hashlib
import sqlite3
import urllib.request
import urllib.error
import urllib.parse
import glob
import html
import html.parser


HTTP_TIMEOUT = 10
CONTEXT_SAFETY_TOKENS = 32
DEFAULT_UNSLOTH_STUDIO_URL = "http://127.0.0.1:8888"
DEFAULT_STUDIO_API_URL = "http://127.0.0.1:8888/api"
RECOVERY_MAX_CYCLES = 3
RELOAD_POLL_SECONDS = 5.0
RELOAD_SETTLE_SECONDS = 300.0
UNLOAD_SETTLE_SECONDS = 120.0
# Port discovery probes dozens/hundreds of stale llama-server ports when old
# logs pile up; a long per-probe timeout turns that into many frozen minutes.
DISCOVERY_PROBE_TIMEOUT = 0.35
DISCOVERY_MAX_PORTS = 4
LLAMA_LOG_FRESH_SECONDS = 6 * 3600
STUDIO_PROBE_TIMEOUT = 1.0
STUDIO_ADMIN_USERNAME = "unsloth"
_URL_PAIR = r"(?:\([^()\s<>\"']*\)|\[[^\[\]\s<>\"']*\])"
URL_PATTERN = re.compile(rf"https?://(?:[^\s<>\"'()\[\]]|{_URL_PAIR})+", re.IGNORECASE)
WEB_FETCH_TIMEOUT = 15
MAX_WEB_CHARS_PER_URL = 12000
MAX_WEB_URLS = 6

MAX_RETAIN_ANSWERS = 16

# Live thinking-display transport. The bridge streams chat completions and
# rebroadcasts partial thinking/answer snapshots over the ComfyUI websocket so
# the companion CRT_UnslothThinkingDisplay node can render them in real time.
CRT_UNSLOTH_THINKING_EVENT = "crt_unsloth_thinking"
STREAM_TIMEOUT = 300
STREAM_BROADCAST_MIN_INTERVAL = 0.25
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_TAIL_RE = re.compile(r"<think>([^<]*)$", re.IGNORECASE)

_WEB_GUARD_SYSTEM_PROMPT = (
    "You are a helpful assistant. Any web content you need was already "
    "fetched and is included in the user message under the heading "
    "'Web content fetched for this request'. Read it and answer from it. "
    "Never claim you cannot access a link, never say a URL is incomplete "
    "or cut off, and never ask the user to paste a link again. If the "
    "message contains a URL, treat it as already retrieved: the content "
    "is embedded in the message itself."
)

_WEB_NOISE_TAGS = frozenset({
    "script", "style", "noscript", "iframe", "object", "embed",
    "head", "form", "nav", "footer", "aside",
})
_WEB_VOID_TAGS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
})
_WEB_BREAK_TAGS = frozenset({
    "p", "div", "section", "article", "blockquote", "pre",
    "ul", "ol", "table", "thead", "tbody", "h1", "h2", "h3", "h4", "h5", "h6", "br",
})


class _PageText(html.parser.HTMLParser):
    """HTML -> clean text. Content region starts at <main>/<article>.
    Blocks that are mostly links (nav, sidebar, TOC, share widgets) are
    dropped by link-density; everything else is kept."""
    CONTENT_TAGS = ("main", "article")
    _LINK_HEAVY_RATIO = 0.65

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip_stack = []
        self._in_content = False
        self._link_depth = 0
        self._heading_depth = 0
        self._blocks = []
        self._cur = []
        self._cur_link = 0
        self._cur_total = 0
        self._pending_break = False

    def _flush(self):
        if self._cur:
            if self._cur_total == 0 or (self._cur_link / self._cur_total) <= self._LINK_HEAVY_RATIO or self._heading_depth > 0:
                self._blocks.append("".join(self._cur))
            self._cur = []
            self._cur_link = 0
            self._cur_total = 0

    def handle_starttag(self, tag, attrs):
        if not self._in_content and tag in self.CONTENT_TAGS:
            self._skip_stack = []
            self._in_content = True
            self._blocks = []
            self._cur = []
            self._cur_link = 0
            self._cur_total = 0
            self._pending_break = False
            return
        if self._skip_stack:
            if tag not in _WEB_VOID_TAGS:
                self._skip_stack.append(tag)
            return
        if not self._in_content:
            return
        lower_attrs = {k.lower(): (v or "").lower() for k, v in attrs}
        cls = lower_attrs.get("class", "")
        if tag in _WEB_NOISE_TAGS or "nav" in cls or ("share" in cls and tag in ("div", "aside")):
            self._skip_stack.append(tag)
            return
        if tag == "a":
            self._link_depth += 1
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._flush()
            self._heading_depth += 1
            self._cur.append("\n" + "# " * int(tag[1]))
            self._cur_total += 1
            self._pending_break = False
        elif tag == "li":
            self._flush()
            self._cur.append("\n- ")
            self._cur_total += 1
            self._pending_break = False
        elif tag in _WEB_BREAK_TAGS:
            self._pending_break = True

    def handle_endtag(self, tag):
        if self._skip_stack:
            if tag in _WEB_VOID_TAGS:
                return
            if self._skip_stack[-1] == tag:
                self._skip_stack.pop()
            return
        if not self._in_content:
            return
        if tag == "a" and self._link_depth:
            self._link_depth -= 1
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6") and self._heading_depth:
            self._heading_depth -= 1
        if tag in ("p", "div", "section", "article", "blockquote", "pre", "ul", "ol", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._pending_break = True

    def handle_data(self, data):
        if not self._in_content or self._skip_stack:
            return
        text = re.sub(r"\s+", " ", data).strip()
        if not text:
            return
        if self._pending_break and self._cur and not self._cur[-1].endswith("\n"):
            self._cur.append("\n")
        self._cur.append(text + " ")
        n = len(text)
        self._cur_total += n
        if self._link_depth > 0 and self._heading_depth == 0:
            self._cur_link += n
        self._pending_break = False

    def text(self):
        self._flush()
        text = "".join(self._blocks)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.fullmatch(r"\s*[-â€“â€”\s]*(Partager|Facebook|Twitter|X|LinkedIn|Copier le lien|WhatsApp|Envoyer|Imprimer|Courriel|E-mail|Lire plus tard|Lire la suite[^\n]*|Lecture\s*:\s*\d+\s*min\.?)\s*", stripped):
                continue
            lines.append(line)
        return "\n".join(lines).strip()


def _request_json(url, body=None, timeout=HTTP_TIMEOUT, retries=1, rediscover_url=None):
    data = None
    method = "GET"
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"
    last_exc = None
    current_url = url
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            current_url,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(payload)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"llama-server returned invalid JSON for {current_url}: {payload[:500]}"
                ) from exc
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace").strip()
            detail = error_body or exc.reason or "no response body"
            raise RuntimeError(
                f"llama-server HTTP {exc.code} for {current_url}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            last_exc = exc
            if attempt < retries:
                # If a rediscover callback is provided, try to find the current active
                # server URL (e.g. Unsloth Studio may have restarted llama-server on a
                # different port). This is not retried on its own; it consumes one attempt.
                if rediscover_url is not None:
                    try:
                        new_url = rediscover_url()
                    except Exception:
                        new_url = None
                    if new_url and new_url != current_url:
                        print(
                            f"[Unsloth Studio Bridge] Retrying with rediscovered server: {new_url}",
                            file=sys.stderr,
                        )
                        current_url = new_url
                        continue
                # Retry transient connection failures such as WinError 10061.
                if isinstance(exc.reason, (ConnectionRefusedError, ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
                    time.sleep(1.0)
                    continue
            raise RuntimeError(f"Could not reach llama-server at {current_url}: {exc.reason}") from exc
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(1.0)
                continue
            raise RuntimeError(
                f"llama-server closed the connection for {current_url}. The server process may "
                "have crashed or restarted; reload the model and retry."
            ) from exc


def _find_latest_server_log():
    unsloth_home = os.path.join(os.environ.get("USERPROFILE", ""), ".unsloth", "studio")
    log_dir = os.path.join(unsloth_home, "logs", "server")
    if not os.path.isdir(log_dir):
        return None
    log_files = glob.glob(os.path.join(log_dir, "server-*.log"))
    if not log_files:
        return None
    log_files.sort(key=os.path.getmtime, reverse=True)
    return log_files[0]


def _find_llama_server_logs():
    unsloth_home = os.path.join(os.environ.get("USERPROFILE", ""), ".unsloth", "studio")
    log_dir = os.path.join(unsloth_home, "logs", "llama-server")
    if not os.path.isdir(log_dir):
        return []
    log_files = glob.glob(os.path.join(log_dir, "llama-*-port-*-try*.log"))
    log_files.sort(key=os.path.getmtime, reverse=True)
    return log_files


def _parse_ports_from_server_log(log_path):
    """Return list of ports in chronological order from server log."""
    if log_path is None:
        return []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return []
    ports = []
    for m in re.finditer(r'--port\s+(\d+)', content):
        ports.append(int(m.group(1)))
    for m in re.finditer(r'ready on port (\d+)', content):
        ports.append(int(m.group(1)))
    return ports


def _parse_port_from_llama_log(log_path):
    if log_path is None:
        return None
    name = os.path.basename(log_path)
    m = re.search(r'-port-(\d+)-', name)
    if m:
        return int(m.group(1))
    return None


def _parse_model_name_from_log(log_path):
    if log_path is None:
        return None
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return None
    m = re.search(r'Detected local GGUF model:.*[\\/]([^\\/]+\.gguf)', content)
    if m:
        return m.group(1)
    m = re.search(r'llama-server ready on port \d+ for model [\'"]([^\'"]+)[\'"]', content)
    if m:
        return m.group(1)
    return None


def _normalize_server_url(value):
    raw = str(value or DEFAULT_UNSLOTH_STUDIO_URL).strip()
    if not raw:
        raw = DEFAULT_UNSLOTH_STUDIO_URL
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            "unsloth_server_url must be an HTTP(S) URL, for example "
            "http://127.0.0.1:8888."
        )
    return raw.rstrip("/")


def _is_server_alive(base_url, timeout=2):
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/health", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _is_port_alive(port, timeout=2):
    return _is_server_alive(f"http://127.0.0.1:{port}", timeout=timeout)


def _resolve_llama_server(configured_url):
    configured_base = _normalize_server_url(configured_url)
    default_base = _normalize_server_url(DEFAULT_UNSLOTH_STUDIO_URL)

    if configured_base != default_base and _is_server_alive(configured_base):
        return configured_base, "configured llama-server URL"

    llama_port = _detect_llama_port()
    if llama_port is None:
        if configured_base != default_base:
            raise RuntimeError(
                f"Could not reach the configured llama-server at {configured_base}, "
                "and no active port could be detected from Unsloth Studio logs."
            )
        raise RuntimeError(
            "Could not detect llama-server port from Unsloth Studio logs. "
            "Make sure Unsloth Studio is running and a model is loaded."
        )

    parsed = urllib.parse.urlparse(configured_base)
    hostname = parsed.hostname or "127.0.0.1"
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    detected_base = f"{parsed.scheme}://{hostname}:{llama_port}"
    if not _is_server_alive(detected_base):
        raise RuntimeError(
            f"llama-server at {detected_base} is not responding. The model may "
            "have been unloaded or the port changed; reload it in Unsloth Studio."
        )
    return detected_base, f"Studio log discovery (port {llama_port})"


def _detect_llama_port():
    unsloth_home = os.path.join(os.environ.get("USERPROFILE", ""), ".unsloth", "studio")
    now = time.time()

    # Strategy 1: check ports from latest server log in reverse chronological order.
    # Only the newest few: each dead port costs a probe timeout, and old logs
    # can carry hundreds of stale ports from previous launches.
    server_log = _find_latest_server_log()
    server_ports = _parse_ports_from_server_log(server_log)
    candidates = []
    for port in reversed(server_ports):
        if port not in candidates:
            candidates.append(port)
        if len(candidates) >= DISCOVERY_MAX_PORTS:
            break
    for port in candidates:
        if _is_port_alive(port, timeout=DISCOVERY_PROBE_TIMEOUT):
            return port

    # Strategy 2: only llama-server logs from the last few hours; the list is
    # sorted newest first, so the first stale file ends the scan.
    fresh = list(candidates)
    for log_path in _find_llama_server_logs():
        if now - os.path.getmtime(log_path) > LLAMA_LOG_FRESH_SECONDS:
            break
        port = _parse_port_from_llama_log(log_path)
        if not port or port in fresh:
            continue
        fresh.append(port)
        if len(fresh) >= DISCOVERY_MAX_PORTS * 2:
            break
        if _is_port_alive(port, timeout=DISCOVERY_PROBE_TIMEOUT):
            return port

    # Strategy 3: fallback to last port from server log even if not alive
    if server_ports:
        return server_ports[-1]
    return None


def _discover_studio_api_url():
    server_log = _find_latest_server_log()
    if server_log is not None:
        try:
            with open(server_log, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            content = ""
        m = re.search(r"Unsloth Studio running on\s+(https?://[^\s\[]+)", content)
        if m:
            return m.group(1).rstrip("/") + "/api"
    return DEFAULT_STUDIO_API_URL


def _read_auth_file(name):
    auth_dir = os.path.join(os.environ.get("USERPROFILE", ""), ".unsloth", "studio", "auth")
    try:
        with open(os.path.join(auth_dir, name), "r", encoding="utf-8") as f:
            value = f.read().strip()
        return value or None
    except Exception:
        return None


_STUDIO_TOKEN = None
_LAST_MODEL_PATH = None


def _mint_studio_token(username):
    """Mint an HS256 JWT the same way Studio does, using the signing secret
    stored in Studio's local auth.db. This mirrors Studio's own desktop auth
    (same user, same machine), so no password input is required."""
    db_path = os.path.join(
        os.environ.get("USERPROFILE", ""), ".unsloth", "studio", "auth", "auth.db"
    )
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        try:
            row = conn.execute(
                "SELECT jwt_secret FROM auth_user WHERE username = ?", (username,)
            ).fetchone()
        finally:
            conn.close()
    except Exception:
        return None
    if not row:
        return None

    def b64url(raw):
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    header = b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode("utf-8"))
    payload = b64url(
        json.dumps(
            {"sub": username, "desktop": True, "exp": int(time.time()) + 3600}
        ).encode("utf-8")
    )
    signing_input = f"{header}.{payload}".encode("ascii")
    signature = b64url(hmac.new(row[0].encode("utf-8"), signing_input, hashlib.sha256).digest())
    return f"{header}.{payload}.{signature}"


def _get_studio_token(studio_api_url, username, credential):
    """Studio auth: an sk-unsloth API key passes straight through; otherwise a
    desktop secret file, bootstrap password file, node credential login, then a
    locally minted token from Studio's auth.db. Cached."""
    global _STUDIO_TOKEN
    if credential and credential.strip().startswith("sk-"):
        return credential.strip()
    if _STUDIO_TOKEN:
        return _STUDIO_TOKEN
    candidates = []
    desktop_secret = _read_auth_file(".desktop_secret")
    if desktop_secret:
        candidates.append(("desktop-login", {"secret": desktop_secret}))
    bootstrap = _read_auth_file(".bootstrap_password")
    if bootstrap:
        candidates.append(("login", {"username": username, "password": bootstrap}))
    if credential:
        candidates.append(("login", {"username": username, "password": credential}))
    for endpoint, body in candidates:
        try:
            result = _studio_call(studio_api_url, None, f"/auth/{endpoint}", body, timeout=10)
            token = result.get("access_token")
            if token:
                _STUDIO_TOKEN = token
                return token
        except Exception:
            continue
    token = _mint_studio_token(username)
    if token:
        _STUDIO_TOKEN = token
    return token


def _studio_call(api_url, token, path, body=None, timeout=60):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
        return json.loads(payload) if payload else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip() or exc.reason
        raise RuntimeError(f"Studio API HTTP {exc.code} for {path}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Studio API at {api_url}{path}: {exc.reason}") from exc


def _parse_model_path_from_log(log_path):
    if log_path is None:
        return None
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return None
    # Newest event wins: one Studio session can load several models, and the
    # first match in the file is then a stale path unload/reload would miss.
    for line in reversed(content.splitlines()):
        try:
            event = json.loads(line).get("event", "")
        except Exception:
            event = line
        m = re.search(r"Detected local GGUF model:\s*(.+?\.gguf)", event)
        if m:
            return m.group(1).strip()
    return None


def _find_last_model_path():
    """Last model loaded by Studio, found by scanning all server logs newest
    first (a fresh log may not contain a load event yet)."""
    log_dir = os.path.join(
        os.environ.get("USERPROFILE", ""), ".unsloth", "studio", "logs", "server"
    )
    try:
        logs = sorted(glob.glob(os.path.join(log_dir, "server-*.log")), key=os.path.getmtime, reverse=True)
    except Exception:
        logs = []
    for log_path in logs:
        model_path = _parse_model_path_from_log(log_path)
        if model_path:
            return model_path
    return None


def _split_launch_args(cmd):
    """Tokenize a command line, keeping JSON groups such as
    --chat-template-kwargs {"enable_thinking": true} together."""
    protected = {}

    def protect(match):
        key = f"\x00{len(protected)}\x00"
        protected[key] = match.group(0)
        return key

    cmd = re.sub(r"\{[^{}]*\}", protect, cmd)
    return [protected.get(tok, tok) for tok in cmd.split()]


def _parse_launch_args(tokens):
    """(flag, value) pairs from a launch command; flags followed by a token
    that does not start with '-' consume it as their value."""
    pairs = []
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("-"):
            i += 1
            continue
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
            pairs.append((tok.lstrip("-"), tokens[i + 1]))
            i += 2
        else:
            pairs.append((tok.lstrip("-"), None))
            i += 1
    return pairs


def _find_last_launch_command(model_path):
    """Most recent 'Starting llama-server' command whose -m matches the model,
    scanning all server logs newest first."""
    log_dir = os.path.join(
        os.environ.get("USERPROFILE", ""), ".unsloth", "studio", "logs", "server"
    )
    try:
        logs = sorted(glob.glob(os.path.join(log_dir, "server-*.log")), key=os.path.getmtime, reverse=True)
    except Exception:
        logs = []
    target = model_path.replace("/", "\\").lower()
    for log_path in logs:
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            continue
        for line in reversed(content.splitlines()):
            try:
                event = json.loads(line).get("event", "")
            except Exception:
                event = line
            if not event.startswith("Starting llama-server: "):
                continue
            cmd = event[len("Starting llama-server: "):]
            pairs = _parse_launch_args(_split_launch_args(cmd))
            seen = set()
            duplicate = False
            for flag, value in pairs:
                if flag in seen:
                    duplicate = True
                    break
                seen.add(flag)
            if duplicate:
                continue
            for flag, value in pairs:
                if flag in ("m", "model") and value and value.lower() == target:
                    return cmd
    return None


def _load_request_from_launch(model_path, launch_cmd):
    """Rebuild the Studio /inference/load request from the llama-server launch
    command Studio used, so a reload restores context length, speculative
    decoding, drafter, vision projector and GPU placement."""
    body = {"model_path": model_path}
    if not launch_cmd:
        return body
    flags = {}
    for flag, value in _parse_launch_args(_split_launch_args(launch_cmd)):
        flags.setdefault(flag, []).append(value)
    ints = {}
    for flag in ("c", "ctx-size", "parallel", "batch-size", "ubatch-size",
                 "spec-draft-n-max", "gpu-layers", "ngl", "n-cpu-moe",
                 "fit-ctx", "fit-target", "threads", "cache-ram", "ctx-checkpoints",
                 "spec-ngram-mod-n-match", "spec-ngram-mod-n-min", "spec-ngram-mod-n-max"):
        if flag in flags and flags[flag][0] is not None:
            try:
                ints[flag] = int(flags[flag][0])
            except ValueError:
                pass
    if "c" in ints and ints["c"] > 0:
        body["max_seq_length"] = ints["c"]
    elif "fit-ctx" in ints:
        body["max_seq_length"] = ints["fit-ctx"]
    if "parallel" in ints:
        body["n_parallel"] = ints["parallel"]
    if "batch-size" in ints:
        body["n_batch"] = ints["batch-size"]
    if "ubatch-size" in ints:
        body["n_ubatch"] = ints["ubatch-size"]
    if ("cache-type-k" in flags and "cache-type-v" in flags
            and flags["cache-type-k"][0] == flags["cache-type-v"][0]
            and flags["cache-type-k"][0]):
        body["cache_type_kv"] = flags["cache-type-k"][0]
    if "spec-type" in flags and flags["spec-type"][0]:
        spec = flags["spec-type"][0]
        if spec == "spec-default":
            pass
        elif spec == "ngram-mod,draft-mtp":
            body["speculative_type"] = "mtp+ngram"
        else:
            body["speculative_type"] = spec
    if "spec-draft-n-max" in ints:
        body["spec_draft_n_max"] = ints["spec-draft-n-max"]
    if "fit" in flags and flags["fit"][0] == "off":
        body["gpu_memory_mode"] = "manual"
        if "gpu-layers" in ints:
            body["gpu_layers"] = ints["gpu-layers"]
        elif "ngl" in ints:
            body["gpu_layers"] = ints["ngl"]
    if "n-cpu-moe" in ints:
        body["n_cpu_moe"] = ints["n-cpu-moe"]
    if "tensor-split" in flags and flags["tensor-split"][0]:
        body["tensor_split"] = [float(x) for x in flags["tensor-split"][0].split(",") if x.strip()]
    if "split-mode" in flags and flags["split-mode"][0] == "tensor":
        body["tensor_parallel"] = True
    mapped = {
        "m", "model", "port", "alias", "fit", "fit-ctx", "fit-target", "metrics",
        "slot-save-path", "c", "ctx-size", "parallel", "batch-size", "ubatch-size",
        "cache-type-k", "cache-type-v", "spec-type", "spec-draft-n-max",
        "spec-draft-n-min", "spec-draft-p-min", "spec-draft-p-split",
        "spec-draft-model", "spec-draft-hf", "hf-repo-draft",
        "gpu-layers", "ngl", "n-cpu-moe", "tensor-split", "split-mode",
    }
    rejected = {
        "mm", "mmproj", "mmu", "mmproj-url", "mu", "model-url", "dr", "docker-repo",
        "hf", "hfr", "hf-repo", "hff", "hf-file", "hfv", "hfrv", "hf-repo-v",
        "hffv", "hf-file-v", "hft", "hf-token", "host", "path", "api-prefix",
        "reuse-port", "api-key", "api-key-file", "ssl-key-file", "ssl-cert-file",
        "webui", "no-webui", "ui", "no-ui", "ui-config", "webui-config",
        "ui-config-file", "webui-config-file", "ui-mcp-proxy", "webui-mcp-proxy",
        "no-ui-mcp-proxy", "no-webui-mcp-proxy", "models-dir", "models-preset",
        "models-max", "models-autoload", "no-models-autoload", "embedding",
        "embeddings", "rerank", "reranking", "pooling", "tools", "ag", "agent",
        "no-ag", "no-agent", "tools-runtime", "mcp-servers-config", "mcp-servers-json",
        "cors-origins", "cors-headers", "cors-methods", "cors-credentials",
        "no-cors-credentials", "media-path", "log-file", "log-disable",
        "help", "usage", "version", "list-devices", "cache-list", "completion-bash",
    }
    extras = []
    emitted = set(mapped)
    for flag, value in _parse_launch_args(_split_launch_args(launch_cmd)):
        if flag in mapped or flag in rejected or value is None or flag in emitted:
            continue
        emitted.add(flag)
        extras.append("--" + flag)
        extras.append(value)
    if extras:
        body["llama_extra_args"] = extras
    return body


def _find_launch_for_port(port):
    """Launch command of the llama-server currently bound to the given port,
    scanning all server logs newest first."""
    log_dir = os.path.join(
        os.environ.get("USERPROFILE", ""), ".unsloth", "studio", "logs", "server"
    )
    try:
        logs = sorted(glob.glob(os.path.join(log_dir, "server-*.log")), key=os.path.getmtime, reverse=True)
    except Exception:
        logs = []
    for log_path in logs:
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            continue
        for line in reversed(content.splitlines()):
            try:
                event = json.loads(line).get("event", "")
            except Exception:
                event = line
            if not event.startswith("Starting llama-server: "):
                continue
            cmd = event[len("Starting llama-server: "):]
            for flag, value in _parse_launch_args(_split_launch_args(cmd)):
                if flag == "port":
                    try:
                        if int(value) == port:
                            return cmd
                    except (TypeError, ValueError):
                        pass
    return None


def _remember_launch(base_url):
    global _LAST_LAUNCH_CMD
    try:
        port = urllib.parse.urlparse(base_url).port
    except Exception:
        port = None
    if port:
        cmd = _find_launch_for_port(port)
        if cmd:
            _LAST_LAUNCH_CMD = cmd


_LAST_LAUNCH_CMD = None


def _reload_model_via_studio(studio_api_url, username, credential, model_path):
    if not model_path:
        return False
    token = _get_studio_token(studio_api_url, username, credential)
    if not token:
        print(
            "[Unsloth Studio Bridge] Cannot auto-reload: no Studio credentials available. "
            "Set the studio_api_key input to enable automatic reload.",
            file=sys.stderr,
        )
        return False
    print(
        f"[Unsloth Studio Bridge] Unloading + reloading model via Studio API: {model_path}",
        file=sys.stderr,
    )
    load_body = _load_request_from_launch(
        model_path, _LAST_LAUNCH_CMD or _find_last_launch_command(model_path)
    )
    load_body["force_cancel_active"] = True
    print(
        f"[Unsloth Studio Bridge] Restoring load settings: "
        f"ctx={load_body.get('max_seq_length', 'auto')}, "
        f"parallel={load_body.get('n_parallel', 'default')}, "
        f"spec={load_body.get('speculative_type', 'auto')}, "
        f"kv_cache={load_body.get('cache_type_kv', 'default')}, "
        f"extra_args={len(load_body.get('llama_extra_args', [])) // 2}",
        file=sys.stderr,
    )
    try:
        try:
            _studio_call(
                studio_api_url,
                token,
                "/inference/unload",
                {"model_path": model_path, "force_cancel_active": True},
                timeout=600,
            )
        except Exception as exc:
            print(f"[Unsloth Studio Bridge] Unload skipped: {exc}", file=sys.stderr)
        try:
            _studio_call(
                studio_api_url,
                token,
                "/inference/load",
                load_body,
                timeout=900,
            )
        except RuntimeError as exc:
            if "HTTP 401" in str(exc):
                global _STUDIO_TOKEN
                _STUDIO_TOKEN = None
                token = _get_studio_token(studio_api_url, username, credential)
                if token:
                    _studio_call(
                        studio_api_url,
                        token,
                        "/inference/load",
                        load_body,
                        timeout=900,
                    )
            else:
                raise
        print("[Unsloth Studio Bridge] Model reloaded via Studio API", file=sys.stderr)
        return True
    except Exception as exc:
        print(f"[Unsloth Studio Bridge] Reload failed: {exc}", file=sys.stderr)
        return False


def _active_model_identifier(studio_api_url, token):
    """Model currently resident in Studio, straight from the API."""
    try:
        status = _studio_call(studio_api_url, token, "/inference/status")
    except Exception:
        return None
    for key in ("model_identifier", "active_model"):
        value = status.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _unload_model_via_studio(studio_api_url, username, credential, model_path, base_url=None):
    global _STUDIO_TOKEN
    token = _get_studio_token(studio_api_url, username, credential)
    if not token:
        print(
            "[Unsloth Studio Bridge] Cannot unload: no Studio credentials available. "
            "Set the studio_api_key input to enable unload after run.",
            file=sys.stderr,
        )
        return False
    # Prefer what the API reports as resident: a log-derived path can be stale
    # after switching models, and Studio skips teardown on a mismatched name.
    active = _active_model_identifier(studio_api_url, token)
    if active:
        model_path = active
    elif not model_path:
        print(
            "[Unsloth Studio Bridge] Cannot unload: Studio reports no loaded model.",
            file=sys.stderr,
        )
        return False
    body = {"model_path": model_path, "force_cancel_active": True}
    try:
        _studio_call(studio_api_url, token, "/inference/unload", body, timeout=600)
    except Exception as exc:
        if "HTTP 401" not in str(exc):
            print(f"[Unsloth Studio Bridge] WARNING: unload failed: {exc}", file=sys.stderr)
            return False
        _STUDIO_TOKEN = None
        token = _get_studio_token(studio_api_url, username, credential)
        if not token:
            return False
        try:
            _studio_call(studio_api_url, token, "/inference/unload", body, timeout=600)
        except Exception as exc:
            print(f"[Unsloth Studio Bridge] WARNING: unload failed: {exc}", file=sys.stderr)
            return False
    print(
        f"[Unsloth Studio Bridge] Model unloaded from VRAM via Studio API: {model_path}",
        file=sys.stderr,
    )
    if base_url:
        deadline = time.monotonic() + UNLOAD_SETTLE_SECONDS
        while time.monotonic() < deadline and _is_server_alive(base_url):
            time.sleep(2.0)
        if _is_server_alive(base_url):
            print(
                "[Unsloth Studio Bridge] WARNING: llama-server still answers "
                "after unload; VRAM may still be held.",
                file=sys.stderr,
            )
    return True


def _studio_reachable(studio_api_url):
    """True when the Studio backend answers at all; auth state irrelevant."""
    try:
        req = urllib.request.Request(
            f"{studio_api_url.rstrip('/')}/inference/status", method="GET"
        )
        with urllib.request.urlopen(req, timeout=STUDIO_PROBE_TIMEOUT):
            return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def _recover_llama_server(configured_url, studio_api_url, username, credential):
    """Reload the last loaded model via Studio when llama-server is down
    (e.g. unloaded after the previous run), then wait for it to answer.
    The load is issued once per cycle and never force-cancelled by a newer
    attempt; returns a live base URL or None when exhausted. Fails within
    seconds when Unsloth Studio itself is not running."""
    global _LAST_MODEL_PATH
    model_path = _LAST_MODEL_PATH or _find_last_model_path()
    if not _studio_reachable(studio_api_url):
        print(
            f"[Unsloth Studio Bridge] Unsloth Studio is not reachable at "
            f"{studio_api_url}; cannot reload the model.",
            file=sys.stderr,
        )
        raise RuntimeError(
            f"llama-server is down and Unsloth Studio is not reachable at "
            f"{studio_api_url}. Start Unsloth Studio and retry."
        )
    for cycle in range(1, RECOVERY_MAX_CYCLES + 1):
        try:
            new_base, source = _resolve_llama_server(configured_url)
            if _is_server_alive(new_base):
                print(
                    f"[Unsloth Studio Bridge] Server recovered: {new_base} ({source})",
                    file=sys.stderr,
                )
                _remember_launch(new_base)
                return new_base
        except Exception:
            pass
        if not _studio_reachable(studio_api_url):
            print(
                f"[Unsloth Studio Bridge] Unsloth Studio is not reachable at "
                f"{studio_api_url}; cannot reload the model.",
                file=sys.stderr,
            )
            raise RuntimeError(
                f"llama-server is down and Unsloth Studio is not reachable at "
                f"{studio_api_url}. Start Unsloth Studio and retry."
            )
        print(
            f"[Unsloth Studio Bridge] llama-server down; loading "
            f"{model_path or 'last model'} via Studio "
            f"(cycle {cycle}/{RECOVERY_MAX_CYCLES})...",
            file=sys.stderr,
        )
        _reload_model_via_studio(studio_api_url, username, credential, model_path)
        deadline = time.monotonic() + RELOAD_SETTLE_SECONDS
        next_beat = 0.0
        while time.monotonic() < deadline:
            try:
                new_base, source = _resolve_llama_server(configured_url)
                if _is_server_alive(new_base):
                    print(
                        f"[Unsloth Studio Bridge] Server recovered: {new_base} ({source})",
                        file=sys.stderr,
                    )
                    _remember_launch(new_base)
                    return new_base
            except Exception:
                pass
            waited = int(time.monotonic() - (deadline - RELOAD_SETTLE_SECONDS))
            if waited >= next_beat:
                print(
                    f"[Unsloth Studio Bridge] Waiting for llama-server... {waited}s",
                    file=sys.stderr,
                )
                next_beat = waited + 30
            time.sleep(RELOAD_POLL_SECONDS)
    print(
        f"[Unsloth Studio Bridge] llama-server did not recover after "
        f"{RECOVERY_MAX_CYCLES} cycles",
        file=sys.stderr,
    )
    return None


def _find_inference_defaults(log_path):
    if log_path:
        unsloth_home = os.path.dirname(os.path.dirname(os.path.dirname(log_path)))
        candidate = os.path.join(
            unsloth_home,
            "unsloth_studio", "Lib", "site-packages", "studio", "backend", "assets", "configs",
            "inference_defaults.json",
        )
        if os.path.isfile(candidate):
            return candidate
    unsloth_home = os.path.join(os.environ.get("USERPROFILE", ""), ".unsloth", "studio")
    candidate = os.path.join(
        unsloth_home,
        "unsloth_studio", "Lib", "site-packages", "studio", "backend", "assets", "configs",
        "inference_defaults.json",
    )
    if os.path.isfile(candidate):
        return candidate
    return None


def _load_inference_defaults(log_path):
    path = _find_inference_defaults(log_path)
    if path is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _match_model_family(model_name, defaults):
    if not model_name:
        return None
    name_lower = model_name.lower()
    for pattern in defaults.get("patterns", []):
        if pattern.lower() in name_lower:
            return pattern
    return None


def _get_model_params(log_path):
    defaults = _load_inference_defaults(log_path)
    model_name = _parse_model_name_from_log(log_path)
    family = _match_model_family(model_name, defaults)
    params = defaults.get("families", {}).get(family, {}) if family else {}
    return {
        "model_name": model_name,
        "family": family,
        "temperature": params.get("temperature", 0.7),
        "top_p": params.get("top_p", 0.95),
        "top_k": params.get("top_k", 20),
        "min_p": params.get("min_p", 0.0),
        "repetition_penalty": params.get("repetition_penalty", 1.0),
        "presence_penalty": params.get("presence_penalty", 0.0),
    }


def _get_server_state(base_url):
    props = _request_json(f"{base_url}/props")
    generation = props.get("default_generation_settings", {})
    server_params = generation.get("params", {})
    context_length = generation.get("n_ctx")
    try:
        context_length = int(context_length)
    except (TypeError, ValueError):
        context_length = None
    return {
        "model_name": props.get("model_alias"),
        "context_length": context_length,
        "temperature": server_params.get("temperature"),
        "top_p": server_params.get("top_p"),
        "top_k": server_params.get("top_k"),
        "min_p": server_params.get("min_p"),
    }


def _merge_active_params(fallback, active):
    merged = dict(fallback)
    for key in ("temperature", "top_p", "top_k", "min_p"):
        value = active.get(key)
        if value is not None:
            merged[key] = value
    if active.get("model_name"):
        merged["model_name"] = active["model_name"]
    return merged


def _image_tensor_to_data_urls(image):
    if image is None:
        return []
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required to encode image inputs for Unsloth LLM.") from exc

    if not hasattr(image, "detach"):
        raise RuntimeError("Expected ComfyUI IMAGE tensor input.")

    tensor = image.detach().cpu().clamp(0.0, 1.0)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 4:
        raise RuntimeError(
            f"Expected IMAGE tensor with shape [B,H,W,C] or [H,W,C], got {tuple(tensor.shape)}."
        )

    data_urls = []
    for frame in tensor.numpy():
        if frame.ndim != 3 or frame.shape[-1] not in (1, 3, 4):
            raise RuntimeError(
                f"Expected image frame with 1, 3, or 4 channels, got shape {frame.shape}."
            )
        frame = (frame * 255.0).round().clip(0, 255).astype("uint8")
        if frame.shape[-1] == 1:
            pil_image = Image.fromarray(frame[:, :, 0], mode="L").convert("RGB")
        elif frame.shape[-1] == 4:
            rgba = Image.fromarray(frame, mode="RGBA")
            pil_image = Image.new("RGB", rgba.size, (255, 255, 255))
            pil_image.paste(rgba, mask=rgba.getchannel("A"))
        else:
            pil_image = Image.fromarray(frame, mode="RGB")

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        data_urls.append(f"data:image/png;base64,{encoded}")
    return data_urls


def _extract_urls(text):
    if not text:
        return []
    urls = []
    for match in URL_PATTERN.findall(str(text)):
        for part in re.split(r"(?=https?://)|\\[nrt]", match, flags=re.IGNORECASE):
            url = part.strip().rstrip(".,;:!?")
            if url and url not in urls:
                urls.append(url)
    return urls


def _html_to_text(raw):
    parser = _PageText()
    try:
        parser.feed(raw)
        parser.close()
    except Exception:
        parser = None
    text = parser.text() if parser is not None else ""
    if not text:
        raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
        raw = re.sub(r"(?s)<[^>]+>", " ", raw)
        text = html.unescape(re.sub(r"\s+", " ", raw)).strip()
    return re.sub(r"\s+", " ", text).strip() if len(text) < 200 else text


def _http_response_bytes(resp, max_bytes):
    chunks = []
    total = 0
    while True:
        chunk = resp.read(min(65536, max_bytes - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total >= max_bytes:
            break
    return b"".join(chunks)


def _decode_body(raw, encoding, content_type):
    encoding = (encoding or "").lower().strip()
    try:
        if encoding == "gzip":
            import gzip
            return gzip.decompress(raw)
        if encoding in ("deflate", "x-deflate"):
            import zlib
            try:
                return zlib.decompress(raw)
            except zlib.error:
                return zlib.decompress(raw[2:])
        if encoding == "br":
            import brotli
            return brotli.decompress(raw)
    except (OSError, EOFError, zlib.error, ValueError, ImportError):
        pass
    return raw


def _fetch_url_text(url, timeout=WEB_FETCH_TIMEOUT, max_chars=MAX_WEB_CHARS_PER_URL):
    request_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "identity",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Referer": urllib.parse.urlparse(url).scheme + "://" + (urllib.parse.urlparse(url).netloc or "") + "/",
    }
    last_exc = None
    for _attempt in range(2):
        req = urllib.request.Request(url, headers=request_headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                read_cap = MAX_WEB_CHARS_PER_URL * 32
                raw = _decode_body(
                    _http_response_bytes(resp, read_cap),
                    resp.headers.get("Content-Encoding"),
                    resp.headers.get("Content-Type", ""),
                )
                content_type = resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} for {url}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"could not reach {url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError(f"timed out fetching {url} after {timeout}s") from exc
        raw_str = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        is_html = "html" in content_type.lower()
        text = _html_to_text(raw_str) if is_html else re.sub(r"\s+", " ", raw_str).strip()
        if not is_html:
            return text[:max_chars]
        has_challenge = (
            "client challenge" in raw_str.lower()
            or "please enable javascript to proceed" in raw_str.lower()
        )
        if not has_challenge:
            text = text[:max_chars]
            if not text:
                raise RuntimeError(f"no readable text at {url}")
            return text
        last_exc = RuntimeError(f"bot challenge page at {url}")
        request_headers["User-Agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        )
        import time as _time
        _time.sleep(1.0)
    raise last_exc


def _search_duckduckgo(query, max_results=5):
    """DuckDuckGo HTML search. Returns [(title, url), ...], ads filtered."""
    data = urllib.parse.urlencode({"q": query}).encode("utf-8")
    req = urllib.request.Request(
        "https://html.duckduckgo.com/html/",
        data=data,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "identity",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=WEB_FETCH_TIMEOUT) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    results = []
    for href, title in re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', raw, re.S):
        target = None
        if "uddg=" in href:
            query_params = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            target = query_params.get("uddg", [None])[0]
        elif href.startswith("//"):
            target = "https:" + href
        elif href.startswith("http"):
            target = href
        if not target or "y.js" in target or "duckduckgo.com" in target:
            continue
        title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", title)).strip()
        if title and (title, target) not in results:
            results.append((title, target))
    return results[:max_results]


def _web_content_from_prompt(prompt, max_urls=MAX_WEB_URLS, max_chars=MAX_WEB_CHARS_PER_URL):
    """Fetch pages for URLs in the prompt; when none are usable, search the
    web for the prompt text and fetch the top results instead.
    Returns (entries, failures, searched), entries are (title, url, text)."""
    urls = _extract_urls(prompt)[:max_urls]
    entries = []
    failures = []
    for url in urls:
        try:
            text = _fetch_url_text(url, max_chars=max_chars)
        except Exception as exc:
            failures.append(f"{url}: {exc}")
            continue
        entries.append((url, url, text))
    if entries:
        return entries, failures, False
    query = re.sub(r"\s+", " ", prompt or "").strip()
    if not query:
        return [], failures, False
    try:
        results = _search_duckduckgo(query, max_urls)
    except Exception as exc:
        failures.append(f"web search failed: {exc}")
        return [], failures, True
    if not results:
        failures.append("web search returned no results")
        return [], failures, True
    for title, url in results:
        try:
            text = _fetch_url_text(url, max_chars=max_chars)
        except Exception as exc:
            failures.append(f"{url}: {exc}")
            continue
        entries.append((title, url, text))
    return entries, failures, True


def _build_web_section(entries):
    sections = ["## Web content fetched for this request", ""]
    for title, url, text in entries:
        sections.append(f"### Source: {title}\nURL: {url}\n\n{text}")
    return "\n\n".join(sections)


def _fetch_urls_from_prompt(prompt, max_urls=MAX_WEB_URLS):
    urls = _extract_urls(prompt)[:max_urls]
    if not urls:
        return [], []
    fetched = []
    failures = []
    for url in urls:
        try:
            text = _fetch_url_text(url)
        except Exception as exc:
            failures.append(f"{url}: {exc}")
            continue
        fetched.append((url, text))
    return fetched, failures


def _load_skills(skills_path):
    raw = str(skills_path or "").strip()
    if not raw:
        return "", 0
    if not os.path.isdir(raw):
        raise RuntimeError(f"Skills path is not a directory: {raw}")
    md_files = glob.glob(os.path.join(raw, "**", "*.md"), recursive=True)
    md_files = sorted({os.path.normpath(p) for p in md_files if os.path.isfile(p)})
    if not md_files:
        return "", 0
    parts = []
    for path in md_files:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read().strip()
        except Exception as exc:
            print(f"[Unsloth Studio Bridge] WARNING: could not read skill file {path}: {exc}", file=sys.stderr)
            continue
        if not content:
            continue
        rel = os.path.relpath(path, raw)
        parts.append(f"## Skill: {rel}\n\n{content}")
    if not parts:
        return "", 0
    return "# Skills\n\n" + "\n\n".join(parts), len(parts)


def _build_user_content(prompt, image):
    text = prompt.strip() if isinstance(prompt, str) else ""
    image_urls = _image_tensor_to_data_urls(image)
    if not image_urls:
        return text, 0

    content = []
    if text:
        content.append({"type": "text", "text": text})
    for image_url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    return content, len(image_urls)


def _messages_for_token_count(messages):
    text_only = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ):
                    text_parts.append(part["text"])
            message = dict(message)
            message["content"] = "\n".join(text_parts)
        text_only.append(message)
    return text_only


def _count_chat_tokens(base_url, messages, disable_thinking):
    template_body = {
        "messages": messages,
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": not disable_thinking},
    }
    rendered = _request_json(
        f"{base_url}/apply-template",
        template_body,
    ).get("prompt")
    if not isinstance(rendered, str):
        raise RuntimeError("llama-server /apply-template did not return a prompt")
    tokenized = _request_json(
        f"{base_url}/tokenize",
        {
            "content": rendered,
            "add_special": True,
            "parse_special": True,
        },
    )
    tokens = tokenized.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError("llama-server /tokenize did not return a token list")
    return len(tokens)


def _validate_context_budget(
    context_length,
    prompt_tokens,
    safety_tokens=CONTEXT_SAFETY_TOKENS,
):
    if context_length is None or context_length <= 0:
        return None
    available = context_length - prompt_tokens - safety_tokens
    if available < 1:
        raise RuntimeError(
            f"Prompt uses {prompt_tokens} tokens, but the active llama-server context "
            f"is only {context_length} tokens ({safety_tokens} reserved). Reload the "
            "model with a larger context or shorten the prompt."
        )
    return available


def _chat_completion(base_url, messages, seed, params, include_reasoning, disable_thinking, cache_skills=True, configured_url=None, studio_credential=""):
    body = {
        "model": "default",
        "messages": messages,
        "seed": seed,
        "temperature": params.get("temperature"),
        "top_p": params.get("top_p"),
        "top_k": params.get("top_k"),
        "min_p": params.get("min_p"),
        "stream": False,
        "cache_prompt": cache_skills,
    }
    if cache_skills:
        body["id_slot"] = 0
    if disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    body = {k: v for k, v in body.items() if v is not None}
    start = time.perf_counter()

    def _recover_url():
        return _recover_llama_server(
            configured_url,
            _discover_studio_api_url(),
            STUDIO_ADMIN_USERNAME,
            studio_credential,
        )

    result = _request_json(
        f"{base_url}/v1/chat/completions",
        body,
        timeout=300,
        retries=2,
        rediscover_url=_recover_url,
    )
    elapsed = time.perf_counter() - start
    choices = result.get("choices")
    if not choices:
        raise RuntimeError("No response from model")
    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "")
    if not isinstance(content, str):
        content = ""
    if not isinstance(reasoning, str):
        reasoning = ""
    usage = result.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    tokens_cached = result.get("tokens_cached")
    cached_note = f", cached={tokens_cached}" if tokens_cached is not None else ""
    tps = completion_tokens / elapsed if elapsed > 0 else 0.0
    print(
        f"[Unsloth Studio Bridge] Tokens: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}{cached_note}, "
        f"time={elapsed:.2f}s, speed={tps:.2f} tok/s",
        file=sys.stderr,
    )
    if include_reasoning and reasoning:
        if content:
            return content + "\n\n[Reasoning]\n" + reasoning
        return reasoning
    return content


def _broadcast_thinking(source_node_id, thinking, answer, done, error=None, stats=None):
    """Best-effort websocket push for the live thinking display node.

    stats is an optional dict with live/final inference figures: tps,
    completion_tokens, prompt_tokens, total_tokens, elapsed, live (bool)."""
    try:
        from server import PromptServer

        server = PromptServer.instance
        if server is None:
            return
        try:
            sid = server.client_id
        except Exception:
            sid = None
        prompt_id = None
        try:
            from comfy_execution.utils import get_executing_context

            ctx = get_executing_context()
            if ctx is not None:
                prompt_id = ctx.prompt_id
        except Exception:
            prompt_id = None
        if prompt_id is None:
            try:
                prompt_id = getattr(server, "last_prompt_id", None)
            except Exception:
                prompt_id = None
        data = {
            "source_node": str(source_node_id) if source_node_id is not None else None,
            "prompt_id": prompt_id,
            "thinking": thinking or "",
            "answer": answer or "",
            "done": bool(done),
        }
        if isinstance(stats, dict) and stats:
            clean = {}
            for key in ("tps", "completion_tokens", "prompt_tokens",
                        "total_tokens", "elapsed", "live"):
                if stats.get(key) is not None:
                    clean[key] = stats[key]
            if clean:
                data["stats"] = clean
        if error:
            data["error"] = str(error)
        server.send_sync(CRT_UNSLOTH_THINKING_EVENT, data, sid)
    except Exception:
        pass


def _partition_live_text(reasoning, answer):
    """Split streamed text into (thinking, visible_answer) for display.

    Prefers the server's dedicated reasoning_content channel; models that
    inline <think>...</think> blocks into content are split out as well so the
    display node's thinking pane stays meaningful either way."""
    reasoning = reasoning or ""
    answer = answer or ""
    embedded = _THINK_BLOCK_RE.findall(answer)
    if embedded:
        visible = _THINK_BLOCK_RE.sub("", answer)
        tail = ""
    else:
        tail_match = _THINK_OPEN_TAIL_RE.search(answer)
        tail = tail_match.group(1) if tail_match else ""
        visible = _THINK_OPEN_TAIL_RE.sub("", answer) if tail else answer
    parts = []
    if reasoning:
        parts.append(reasoning)
    parts.extend(embedded)
    if tail and tail not in parts:
        parts.append(tail)
    return "".join(parts) if len(parts) == 1 and reasoning else "\n".join(p for p in parts if p), visible


def _extract_stream_delta(obj):
    choices = obj.get("choices")
    if not choices:
        return "", ""
    choice = choices[0] if isinstance(choices, list) else {}
    if not isinstance(choice, dict):
        return "", ""
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        delta = choice.get("message")
    if not isinstance(delta, dict):
        delta = {}
    content = delta.get("content", "")
    reasoning = delta.get("reasoning_content", delta.get("reasoning", ""))
    if not isinstance(content, str):
        content = ""
    if not isinstance(reasoning, str):
        reasoning = ""
    if not content and not reasoning and isinstance(choice.get("text"), str):
        content = choice["text"]
    return reasoning, content


def _chat_completion_stream(base_url, messages, seed, params, disable_thinking,
                            cache_skills=True, configured_url=None,
                            studio_credential="", on_token=None):
    """Streaming chat completion. Returns (content, reasoning, stats).

    on_token is called as on_token(reasoning, content, done, stats) where
    stats carries a live tok/s estimate during streaming and the server's real
    usage figures on the final call. Live tok/s counts one SSE delta chunk as
    one token, which matches llama-server's per-token streaming."""
    body = {
        "model": "default",
        "messages": messages,
        "seed": seed,
        "temperature": params.get("temperature"),
        "top_p": params.get("top_p"),
        "top_k": params.get("top_k"),
        "min_p": params.get("min_p"),
        "stream": True,
        "cache_prompt": cache_skills,
    }
    if cache_skills:
        body["id_slot"] = 0
    if disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    body = {k: v for k, v in body.items() if v is not None}
    start = time.perf_counter()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    content_parts = []
    reasoning_parts = []
    usage = {}
    live_tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=STREAM_TIMEOUT) as resp:
            buf = ""
            while True:
                try:
                    raw = resp.read(4096)
                except TimeoutError as exc:
                    raise RuntimeError(
                        f"llama-server streaming timed out after {STREAM_TIMEOUT}s"
                    ) from exc
                if not raw:
                    break
                buf += raw.decode("utf-8", errors="replace")
                lines = buf.split("\n")
                buf = lines.pop()
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        buf = ""
                        raw = b""
                        break
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and isinstance(obj.get("usage"), dict):
                        usage = obj["usage"]
                    piece_reasoning, piece_content = _extract_stream_delta(obj)
                    if piece_reasoning:
                        reasoning_parts.append(piece_reasoning)
                    if piece_content:
                        content_parts.append(piece_content)
                    if (piece_reasoning or piece_content) and on_token is not None:
                        live_tokens += 1
                        live_elapsed = time.perf_counter() - start
                        try:
                            on_token(
                                "".join(reasoning_parts),
                                "".join(content_parts),
                                False,
                                {
                                    "tps": live_tokens / live_elapsed if live_elapsed > 0 else 0.0,
                                    "completion_tokens": live_tokens,
                                    "elapsed": live_elapsed,
                                    "live": True,
                                },
                            )
                        except Exception:
                            pass
                else:
                    continue
                break
            tail = buf.strip()
            if tail.startswith("data:"):
                payload = tail[5:].strip()
                if payload and payload != "[DONE]":
                    try:
                        obj = json.loads(payload)
                        piece_reasoning, piece_content = _extract_stream_delta(obj)
                        if piece_reasoning:
                            reasoning_parts.append(piece_reasoning)
                        if piece_content:
                            content_parts.append(piece_content)
                    except json.JSONDecodeError:
                        pass
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip() or exc.reason
        raise RuntimeError(
            f"llama-server HTTP {exc.code} for {base_url}/v1/chat/completions: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach llama-server at {base_url}: {exc.reason}"
        ) from exc
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    if not content and not reasoning:
        raise RuntimeError("llama-server streaming returned no content")
    elapsed = time.perf_counter() - start
    completion_tokens = usage.get("completion_tokens", 0) or live_tokens
    prompt_tokens = usage.get("prompt_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    tps = completion_tokens / elapsed if elapsed > 0 and completion_tokens else 0.0
    print(
        f"[Unsloth Studio Bridge] Tokens(stream): prompt={prompt_tokens}, "
        f"completion={completion_tokens}, total={total_tokens}, "
        f"time={elapsed:.2f}s, speed={tps:.2f} tok/s",
        file=sys.stderr,
    )
    stats = {
        "tps": tps,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "elapsed": elapsed,
        "live": False,
    }
    if on_token is not None:
        try:
            on_token(reasoning, content, True, stats)
        except Exception:
            pass
    return content, reasoning, stats


class UnslothLLM:
    def __init__(self):
        # Per-node conversation cache: ComfyUI reuses the class instance of each
        # node across runs, so this history is scoped to this node's session.
        self._answer_cache = []

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # NaN != NaN, so every queue hits the server: required for
        # retain_last_response history and the unload/reload cycle to advance.
        return float("NaN")

    def _retained_turns(self):
        return list(self._answer_cache)

    def _remember_turn(self, user_content, response, retain):
        # Only the assistant answers are retained â€” the system prompt /
        # instruction doesn't change, so keeping the questions would just
        # fill the context with duplicates.
        self._answer_cache.append({"role": "assistant", "content": response})
        while len(self._answer_cache) > retain:
            self._answer_cache.pop(0)

    def _retain_history(self, retain, messages):
        turns = self._retained_turns()
        if retain > 0 and turns:
            messages.extend(turns)
            print(
                f"[Unsloth Studio Bridge] Retaining last {len(turns)} answer(s)",
                file=sys.stderr,
            )
        return turns

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Optional image batch. The model loaded in Unsloth Studio must support vision input.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": (
                            "Sampling temperature sent in the request. -1 = use the "
                            "server's active default. Set >= 0 to override per-run; "
                            "this does NOT change Unsloth Studio or its UI."
                        ),
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Top-P (nucleus) sampling sent in the request. -1 = use the "
                            "server's active default. Set >= 0 to override per-run; "
                            "this does NOT change Unsloth Studio or its UI."
                        ),
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 10000,
                        "step": 1,
                        "tooltip": (
                            "Top-K sampling sent in the request. -1 = use the server's "
                            "active default. 0 = unlimited. Set >= 0 to override per run."
                        ),
                    },
                ),
                "unsloth_server_url": (
                    "STRING",
                    {
                        "default": DEFAULT_UNSLOTH_STUDIO_URL,
                        "multiline": False,
                        "tooltip": (
                            "Leave the default to discover Studio's active local llama-server "
                            "port from its logs. Enter a reachable non-default llama-server "
                            "base URL to bypass discovery. Studio must have a model loaded; "
                            "image input requires a vision-capable model."
                        ),
                    },
                ),
                "skills_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Optional directory with skill files. Every .md file in the folder "
                            "and its subfolders is loaded recursively and appended to the "
                            "system prompt as extra context."
                        ),
                    },
                ),
                "disable_thinking": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Request a direct answer by disabling supported model thinking mode. Models that ignore this chat-template option are unaffected.",
                    },
                ),
                "include_reasoning": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Append reasoning_content returned by the server to the response output when the loaded model provides it.",
                    },
                ),
                "retain_last_response": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_RETAIN_ANSWERS,
                        "step": 1,
                        "tooltip": (
                            "Number of previous assistant answers to keep in context for the next run. "
                            "Only the answers are retained â€” the instruction / system prompt is not "
                            "duplicated. 0 is stateless. Set to N to keep the last N answers in "
                            "history so the model can build on its previous outputs."
                        ),
                    },
                ),
                "cache_skills": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Keep the skill files in the llama-server slot "
                            "cache between runs: only the new user content (image + prompt) is "
                            "evaluated, making repeated runs much faster. Editing the system "
                            "prompt or a skill file changes the prefix and automatically "
                            "rebuilds the cache once."
                        ),
                    },
                ),
                "disable_web_search": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Disable automatic URL fetching and web search. When enabled, "
                            "URLs in the prompt are ignored and no web content is fetched."
                        ),
                    },
                ),
                "url_web_search": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "Optional joined URL string to fetch (e.g. from a join/text node, "
                            "one link per line). Socket-only: no frontend text processing applies."
                        ),
                    },
                ),
                "studio_api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Optional. Paste your Unsloth Studio API key (sk-unsloth-...) "
                            "or your Studio login password. Leave empty to authenticate "
                            "automatically on this machine. Used for unload after run "
                            "and auto-reload of the last model."
                        ),
                    },
                ),
                "unload_model_after_run": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Unload the model from Unsloth Studio after each completed run, "
                            "freeing its VRAM for other tasks. The next run automatically "
                            "reloads it via the Studio API with its previous settings "
                            "(requires Studio auth: leave studio_api_key empty for local "
                            "auto-auth, or paste an sk-unsloth key)."
                        ),
                    },
                ),
                "live_display": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Stream thinking + answer tokens live to the companion "
                            "'Thinking Display (CRT)' node via websocket. Turn off to "
                            "use the original single-shot request with no live updates."
                        ),
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "CRT/LLM"
    DESCRIPTION = (
        "Connects ComfyUI to the model currently loaded in Unsloth Studio. "
        "Unsloth Studio must remain open; image input requires a vision model. "
        "Detects http(s) URLs in the prompt and url_web_search inputs and fetches their content. "
        "Set retain_last_response above 0 to keep previous answers in context across runs. "
        "Set temperature / top_p / top_k to -1 to use the server's active defaults, or to any value "
        "to override per run. Enable unload_model_after_run to release the model's VRAM "
        "after each run; the next run reloads it automatically."
    )

    def generate(self, prompt, seed, unsloth_server_url=DEFAULT_UNSLOTH_STUDIO_URL, skills_path="", disable_thinking=True, include_reasoning=False, cache_skills=True, retain_last_response=0, image=None, temperature=-1.0, top_p=-1.0, top_k=-1, disable_web_search=False, url_web_search="", studio_api_key="", unload_model_after_run=False, live_display=True, unique_id=None):
        if (not prompt or not prompt.strip()) and image is None:
            print("[Unsloth Studio Bridge][WARN] Prompt is empty", file=sys.stderr)
        print(
            f"[Unsloth Studio Bridge][INFO] Request prepared: "
            f"prompt_chars={len(prompt or '')}, image_connected={image is not None}, retain={retain_last_response}",
            file=sys.stderr,
        )
        studio_api_url = _discover_studio_api_url()
        try:
            llama_base, connection_source = _resolve_llama_server(unsloth_server_url)
        except Exception:
            llama_base = _recover_llama_server(
                unsloth_server_url,
                studio_api_url,
                STUDIO_ADMIN_USERNAME,
                studio_api_key,
            )
            if llama_base is None:
                raise
            connection_source = "recovered after crash"
        _remember_launch(llama_base)
        server_log = _find_latest_server_log()
        global _LAST_MODEL_PATH
        _LAST_MODEL_PATH = _parse_model_path_from_log(server_log) or _LAST_MODEL_PATH
        fallback_params = _get_model_params(server_log)
        server_state = _get_server_state(llama_base)
        params = _merge_active_params(fallback_params, server_state)
        if temperature is not None and temperature >= 0.0:
            params["temperature"] = temperature
        if top_p is not None and top_p >= 0.0:
            params["top_p"] = top_p
        if top_k is not None and top_k >= 0:
            params["top_k"] = top_k
        print(
            f"[Unsloth Studio Bridge][INFO] Connected via {connection_source}: "
            f"{llama_base}, model={params['model_name']}, family={params['family']}",
            file=sys.stderr,
        )
        print(
            f"[Unsloth Studio Bridge] Active server: context={server_state['context_length']}, "
            f"temp={params['temperature']}, top_p={params['top_p']}, "
            f"top_k={params['top_k']}, min_p={params['min_p']} (node overrides: "
            f"temperature={temperature if temperature >= 0.0 else 'server'}, "
            f"top_p={top_p if top_p >= 0.0 else 'server'}, "
            f"top_k={top_k if top_k >= 0 else 'server'})",
            file=sys.stderr,
        )
        messages = []
        skills_text, skills_count = _load_skills(skills_path)
        if skills_count:
            print(f"[Unsloth Studio Bridge] Loaded {skills_count} skill file(s) from {skills_path}", file=sys.stderr)
        system_parts = []
        if skills_text:
            system_parts.append(skills_text)
        if not disable_web_search:
            system_parts.append(_WEB_GUARD_SYSTEM_PROMPT)
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        self._retain_history(retain_last_response, messages)
        user_content, image_count = _build_user_content(prompt, image)
        if image_count:
            print(f"[Unsloth Studio Bridge] Attached {image_count} image(s) to request", file=sys.stderr)
        if disable_web_search:
            print("[Unsloth Studio Bridge] Web search disabled by node toggle", file=sys.stderr)
        else:
            fetched, failures, searched = _web_content_from_prompt(
                f"{prompt or ''}\n{url_web_search or ''}"
            )
            for failure in failures:
                print(f"[Unsloth Studio Bridge] WARNING: {failure}", file=sys.stderr)
            if fetched:
                web_text = _build_web_section(fetched)
                if isinstance(user_content, list):
                    user_content.insert(0, {"type": "text", "text": web_text})
                else:
                    user_content = f"{web_text}\n\n{user_content}"
                if searched:
                    print(
                        f"[Unsloth Studio Bridge] No URL in prompt, searched the web and "
                        f"fetched {len(fetched)} result(s)",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[Unsloth Studio Bridge] Fetched {len(fetched)} URL(s) from prompt: "
                        + ", ".join(url for _, url, _ in fetched),
                        file=sys.stderr,
                    )
            elif failures:
                print(
                    "[Unsloth Studio Bridge] WARNING: web fetch/search produced nothing usable "
                    "(see failures above); sending the raw prompt",
                    file=sys.stderr,
                )
        user_message = {"role": "user", "content": user_content}
        messages.append(user_message)
        prompt_tokens = _count_chat_tokens(
            llama_base,
            _messages_for_token_count(messages),
            disable_thinking,
        )
        available_tokens = _validate_context_budget(
            server_state["context_length"],
            prompt_tokens,
        )
        if image_count:
            print(
                "[Unsloth Studio Bridge] Token budget counts text prompt only; image tokens are handled by llama-server.",
                file=sys.stderr,
            )
        print(
            f"[Unsloth Studio Bridge] Token budget: prompt={prompt_tokens}, "
            f"context={server_state['context_length']}, available={available_tokens}, "
            "completion_limit=server-native",
            file=sys.stderr,
        )
        print(
            f"[Unsloth Studio Bridge] Sending request (seed={seed}, max_tokens=omitted, "
            f"disable_thinking={disable_thinking}, cache_skills={cache_skills}, "
            f"live_display={live_display})",
            file=sys.stderr,
        )
        stream_state = {"last_emit": 0.0}

        def _on_stream_token(stream_reasoning, stream_content, stream_done=False, stream_stats=None):
            if not live_display:
                return
            now = time.monotonic()
            if not stream_done and (now - stream_state["last_emit"]) < STREAM_BROADCAST_MIN_INTERVAL:
                return
            stream_state["last_emit"] = now
            thinking_view, answer_view = _partition_live_text(stream_reasoning, stream_content)
            _broadcast_thinking(unique_id, thinking_view, answer_view, stream_done, stats=stream_stats)

        response = None
        if live_display:
            if unique_id is not None:
                _broadcast_thinking(unique_id, "", "", False)
            try:
                stream_content, stream_reasoning, stream_stats = _chat_completion_stream(
                    llama_base,
                    messages,
                    seed,
                    params,
                    disable_thinking,
                    cache_skills,
                    configured_url=unsloth_server_url,
                    studio_credential=studio_api_key,
                    on_token=_on_stream_token,
                )
                if include_reasoning and stream_reasoning:
                    if stream_content:
                        response = stream_content + "\n\n[Reasoning]\n" + stream_reasoning
                    else:
                        response = stream_reasoning
                else:
                    response = stream_content
                thinking_view, answer_view = _partition_live_text(stream_reasoning, stream_content)
                _broadcast_thinking(unique_id, thinking_view, answer_view, True, stats=stream_stats)
            except Exception as exc:
                print(
                    f"[Unsloth Studio Bridge] WARNING: streaming failed ({exc}); "
                    "falling back to single-shot request.",
                    file=sys.stderr,
                )
                response = None
        if response is None:
            response = _chat_completion(
                llama_base,
                messages,
                seed,
                params,
                include_reasoning,
                disable_thinking,
                cache_skills,
                configured_url=unsloth_server_url,
                studio_credential=studio_api_key,
            )
            if live_display:
                if include_reasoning and "[Reasoning]" in response:
                    head, _, tail = response.partition("[Reasoning]")
                    _broadcast_thinking(unique_id, tail.strip(), head.strip(), True)
                else:
                    thinking_view, answer_view = _partition_live_text("", response)
                    _broadcast_thinking(unique_id, thinking_view, answer_view, True)
        print(
            f"[Unsloth Studio Bridge][INFO] Response received: {len(response)} chars",
            file=sys.stderr,
        )
        if retain_last_response > 0:
            self._remember_turn(user_content, response, retain_last_response)
            print(
                f"[Unsloth Studio Bridge] Retained {retain_last_response} answer(s) in cache: {len(self._answer_cache) // 2} kept",
                file=sys.stderr,
            )
        if unload_model_after_run:
            _unload_model_via_studio(
                studio_api_url,
                STUDIO_ADMIN_USERNAME,
                studio_api_key,
                _LAST_MODEL_PATH or _find_last_model_path(),
                base_url=llama_base,
            )
        return (response,)


NODE_CLASS_MAPPINGS = {
    "UnslothLLM": UnslothLLM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnslothLLM": "Unsloth Studio Bridge (CRT)",
}
