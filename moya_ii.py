# -*- coding: utf-8 -*-
"""
Моя ИИ — локальное приложение с ИИ, работает на видеокарте.
Без Ollama, без API-ключей, после первой загрузки — без интернета.

Как это работает:
  * при первом запуске приложение само скачивает «движок» llama.cpp (Vulkan) —
    он умеет считать на видеокарте (NVIDIA, AMD, Intel);
  * модели (.gguf) скачиваются кнопкой «Скачать модель» в папку modeli;
  * старые модели-llamafile из папки modeli тоже подхватываются:
    из них один раз достаётся .gguf, и они начинают работать на видеокарте.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
import zipfile
import tkinter as tk
from tkinter import ttk, scrolledtext

# ===================== НАСТРОЙКИ =====================
AI_NAME = "Нейра"             # имя твоей ИИ
SYSTEM_PROMPT = (
    f"Тебя зовут {AI_NAME}. Ты дружелюбный и умный помощник. "
    "Отвечай на русском языке, понятно и по делу. "
    "Пиши обычным текстом, без таблиц и без разметки."
)
MEMORY_SIZE = 12              # сколько последних сообщений помнить (меньше = быстрее)
CONTEXT = 4096                # «рабочая память» модели в токенах (меньше = экономнее)
# =====================================================

if getattr(sys, "frozen", False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, "modeli")
ENGINE_DIR = os.path.join(APP_DIR, "dvizhok")
HISTORY_FILE = os.path.join(APP_DIR, "pamyat.json")
IS_WIN = os.name == "nt"
NO_WINDOW = 0x08000000 if IS_WIN else 0
UA = {"User-Agent": "MoyaII/3.0"}

ENGINE_NAME = "llama-server.exe" if IS_WIN else "llama-server"
GH_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=40"


def is_vulkan_asset(name):
    """Подходит ли файл выпуска: Windows + Vulkan + 64 бит, zip."""
    n = name.lower()
    return ("win" in n and "vulkan" in n and "arm64" not in n
            and n.endswith(".zip"))

# модели для скачивания: (название, репозиторий, файл, описание)
CATALOG = [
    ("Быстрая — Qwen3.5 2B (1.3 ГБ)", "unsloth/Qwen3.5-2B-GGUF",
     "Qwen3.5-2B-UD-Q4_K_XL.gguf", "летает на видеокарте, но попроще"),
    ("Умная — Qwen3.5 4B (2.7 ГБ)  ★ советую", "unsloth/Qwen3.5-4B-GGUF",
     "Qwen3.5-4B-Q4_K_M.gguf", "намного умнее, целиком влезает в 4 ГБ видеопамяти"),
]
DEFAULT_CHOICE = 1


# ---------- общие помощники ----------
def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def server_ready(base):
    try:
        with urllib.request.urlopen(base + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def download_file(url, dest, progress=None):
    """Скачивает файл с докачкой. progress(получено, всего, скорость_МБ/с)."""
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = dict(UA)
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        if have and resp.status != 206:
            have = 0
        total = int(resp.headers.get("Content-Length") or 0) + have
        got, t0, last = have, time.time(), 0
        with open(part, "ab" if have else "wb") as f:
            while True:
                block = resp.read(1024 * 1024)
                if not block:
                    break
                f.write(block)
                got += len(block)
                now = time.time()
                if progress and now - last > 0.3:
                    last = now
                    progress(got, total, (got - have) / max(now - t0, 0.1) / 1e6)
    if total and got < total:
        raise RuntimeError("загрузка прервалась")
    os.replace(part, dest)


# ---------- движок ----------
def find_engine():
    if not os.path.isdir(ENGINE_DIR):
        return None
    for root, _dirs, files in os.walk(ENGINE_DIR):
        if ENGINE_NAME in files:
            return os.path.join(root, ENGINE_NAME)
    return None


# ---------- модели ----------
def stem_of(name):
    low = name.lower()
    for ext in (".llamafile.exe", ".llamafile", ".gguf", ".exe"):
        if low.endswith(ext):
            return name[: -len(ext)]
    return name


def find_models():
    """Возвращает список моделей: {'name', 'gguf', 'lf'}."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    found = {}
    for name in sorted(os.listdir(MODELS_DIR)):
        path = os.path.join(MODELS_DIR, name)
        if not os.path.isfile(path):
            continue
        low = name.lower()
        if low.endswith(".part") or "mmproj" in low:
            continue
        kind = None
        if low.endswith(".gguf"):
            kind = "gguf"
        elif low.endswith(".llamafile") or (low.endswith(".exe")
                                           and os.path.getsize(path) > 100 * 1024 * 1024):
            kind = "lf"
        if kind:
            st = stem_of(name)
            found.setdefault(st, {"name": st, "gguf": None, "lf": None})[kind] = path
    return list(found.values())


def extract_gguf(lf_path, dest, progress=None):
    """Достаёт .gguf из файла llamafile (это zip-архив внутри)."""
    with zipfile.ZipFile(lf_path) as z:
        members = [m for m in z.infolist()
                   if m.filename.lower().endswith(".gguf") and "mmproj" not in m.filename.lower()]
        if not members:
            raise RuntimeError("внутри не нашлось модели .gguf")
        m = max(members, key=lambda x: x.file_size)
        part = dest + ".part"
        done = 0
        with z.open(m) as src, open(part, "wb") as out:
            while True:
                block = src.read(4 * 1024 * 1024)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if progress:
                    progress(done, m.file_size)
    os.replace(part, dest)


# ---------- память ----------
def load_history():
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history[-MEMORY_SIZE:], f, ensure_ascii=False, indent=1)
    except Exception:
        pass


class ThinkFilter:
    """Прячет «размышления» модели между <think> и </think>."""
    def __init__(self):
        self.inside = False
        self.buf = ""

    def feed(self, text):
        self.buf += text
        out = ""
        while True:
            tag = "</think>" if self.inside else "<think>"
            i = self.buf.find(tag)
            if i == -1:
                keep = len(tag) - 1
                safe = self.buf[:-keep] if len(self.buf) > keep else ""
                if not self.inside:
                    out += safe
                self.buf = self.buf[len(safe):]
                return out
            if not self.inside:
                out += self.buf[:i]
            self.buf = self.buf[i + len(tag):]
            self.inside = not self.inside

    def flush(self):
        rest = "" if self.inside else self.buf
        self.buf = ""
        return rest


class MarkdownStream:
    """Превращает разметку модели в нормальный текст прямо во время печати:
    **жирный** -> жирный шрифт, «* пункт» -> «• пункт», «### Заголовок» -> жирная строка."""
    def __init__(self):
        self.buf = ""
        self.bold = False
        self.line_start = True
        self.header = False
        self.prev = " "

    def _tag(self):
        return "bold" if (self.bold or self.header) else None

    def feed(self, text, final=False):
        self.buf += text
        out = []           # список (текст, тег)
        b, i = self.buf, 0

        def emit(s):
            tag = self._tag()
            if out and out[-1][1] == tag:
                out[-1] = (out[-1][0] + s, tag)
            else:
                out.append((s, tag))

        while i < len(b):
            rest = len(b) - i
            if self.line_start:
                if b[i] == "#":
                    j = i
                    while j < len(b) and b[j] == "#":
                        j += 1
                    if j >= len(b) and not final:
                        break                      # ждём продолжения
                    if j < len(b) and b[j] == " ":
                        self.header = True
                        i = j + 1
                        self.line_start = False
                        continue
                if b[i] in "*-" and (rest >= 2 or final):
                    if rest >= 2 and b[i + 1] == " ":
                        emit("• ")
                        i += 2
                        self.line_start = False
                        continue
                elif b[i] in "*-" and not final:
                    break
            ch = b[i]
            if ch == "*":
                if rest >= 2 and b[i + 1] == "*":
                    self.bold = not self.bold
                    i += 2
                    self.line_start = False
                    continue
                if rest < 2 and not final:
                    break                          # вдруг дальше вторая звёздочка
                nxt = b[i + 1] if rest >= 2 else " "
                if (self.prev.isdigit() or self.prev == " ") and (nxt.isdigit() or nxt == " "):
                    emit("*")                      # это умножение, а не разметка
                    self.prev = "*"
                i += 1                             # одиночную * (курсив) убираем
                continue
            if ch == "\n":
                self.header = False
                self.bold = False
                emit("\n")
                self.line_start = True
            else:
                emit(ch)
                if ch not in " \t":
                    self.line_start = False
            self.prev = ch
            i += 1
        self.buf = b[i:]
        return out

    def flush(self):
        return self.feed("", final=True)


# ---------- приложение ----------
class ChatApp:
    def __init__(self, root):
        self.root = root
        self.history = load_history()
        self.busy = True
        self.downloading = False
        self.proc = None
        self.base = None
        self.engine = find_engine()
        self.models = []

        root.title(f"{AI_NAME} — моя ИИ")
        root.geometry("780x660")
        root.configure(bg="#1e1f24")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        top = tk.Frame(root, bg="#1e1f24")
        top.pack(fill=tk.X, padx=10, pady=(10, 0))
        tk.Label(top, text="Модель:", bg="#1e1f24", fg="#cccccc",
                 font=("Segoe UI", 10)).pack(side=tk.LEFT)
        self.combo = ttk.Combobox(top, state="readonly", width=40)
        self.combo.pack(side=tk.LEFT, padx=6)
        self.combo.bind("<<ComboboxSelected>>", self.on_model_change)
        tk.Button(top, text="⬇ Скачать модель", command=self.open_download_window,
                  bg="#16a34a", fg="white", relief=tk.FLAT).pack(side=tk.LEFT)
        self.status = tk.Label(top, text="", bg="#1e1f24", fg="#888888",
                               font=("Segoe UI", 9))
        self.status.pack(side=tk.RIGHT)

        self.chat = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, state=tk.DISABLED, font=("Segoe UI", 11),
            bg="#26272e", fg="#e8e8ea", relief=tk.FLAT, padx=12, pady=10)
        self.chat.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        self.chat.tag_config("you", foreground="#7cc4ff", font=("Segoe UI", 11, "bold"))
        self.chat.tag_config("ai", foreground="#9be59b", font=("Segoe UI", 11, "bold"))
        self.chat.tag_config("info", foreground="#888888", font=("Segoe UI", 9, "italic"))
        self.chat.tag_config("bold", font=("Segoe UI", 11, "bold"))

        bottom = tk.Frame(root, bg="#1e1f24")
        bottom.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.entry = tk.Text(bottom, height=3, font=("Segoe UI", 11), wrap=tk.WORD,
                             bg="#2e2f37", fg="white", insertbackground="white",
                             relief=tk.FLAT)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.entry.bind("<Return>", self.on_enter)
        self.entry.focus()
        btns = tk.Frame(bottom, bg="#1e1f24")
        btns.pack(side=tk.RIGHT)
        tk.Button(btns, text="Отправить", command=self.send, width=12,
                  bg="#3b82f6", fg="white", relief=tk.FLAT).pack(pady=(0, 4))
        tk.Button(btns, text="Забыть всё", command=self.clear_memory, width=12,
                  bg="#444", fg="white", relief=tk.FLAT).pack()

        threading.Thread(target=self.startup, daemon=True).start()

    # ----- вывод -----
    def write(self, text, tag=None):
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(tk.END, text, tag)
        self.chat.configure(state=tk.DISABLED)
        self.chat.see(tk.END)

    def say(self, text, tag=None):
        self.root.after(0, self.write, text, tag)

    def set_status(self, text):
        self.root.after(0, lambda: self.status.config(text=text))

    # ----- запуск -----
    def startup(self):
        if IS_WIN and not self.engine:
            self.say("(Первый запуск: скачиваю движок для видеокарты, ~50 МБ...)\n", "info")
            self.set_status("скачиваю движок...")
            try:
                self.engine = self.download_engine()
                self.say("(Движок готов!)\n", "info")
            except Exception as e:
                self.say(f"[Не удалось скачать движок: {e}. Проверь интернет и "
                         "перезапусти. Пока модели-llamafile будут работать на процессоре.]\n",
                         "info")
        self.root.after(0, self.after_startup)

    def download_engine(self):
        req = urllib.request.Request(GH_RELEASES,
                                     headers=dict(UA, Accept="application/vnd.github+json"))
        with urllib.request.urlopen(req, timeout=30) as r:
            releases = json.loads(r.read().decode("utf-8"))
        asset = None
        # самый свежий выпуск, в котором уже есть сборка под Vulkan
        for rel in releases:
            if rel.get("draft"):
                continue
            asset = next((a for a in rel.get("assets", []) if is_vulkan_asset(a["name"])), None)
            if asset:
                break
        if not asset:
            seen = [f'{r.get("tag_name")}({len(r.get("assets", []))})' for r in releases[:8]]
            raise RuntimeError("не нашёл сборку под Vulkan. Выпуски: " + ", ".join(seen))
        os.makedirs(ENGINE_DIR, exist_ok=True)
        zpath = os.path.join(ENGINE_DIR, "engine.zip")
        marks = set()

        def prog(got, total, _speed):
            if total:
                step = int(got * 100 / total) // 25 * 25
                if step not in marks:
                    marks.add(step)
                    self.say(f"  движок: {step}%\n", "info")
        download_file(asset["browser_download_url"], zpath, prog)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(ENGINE_DIR)
        os.remove(zpath)
        path = find_engine()
        if not path:
            raise RuntimeError("в архиве нет llama-server")
        return path

    def after_startup(self):
        self.reload_models()
        if self.models:
            self.combo.current(0)
            self.start_model(self.models[0])
        else:
            self.write("Привет! Чтобы я заработала, нужно один раз скачать мой «мозг» — "
                       "модель. Выбери её в окошке.\n\n", "info")
            self.status.config(text="нет моделей")
            self.root.after(300, self.open_download_window)

    def reload_models(self):
        self.models = find_models()
        if not self.engine:   # без движка работают только llamafile
            self.models = [m for m in self.models if m["lf"]]
        self.combo.config(values=[m["name"] for m in self.models])

    # ----- окно скачивания -----
    def open_download_window(self):
        if self.downloading:
            return
        if not self.engine:
            self.write("[Сначала нужен движок для видеокарты — перезапусти приложение "
                       "с включённым интернетом.]\n", "info")
            return
        win = tk.Toplevel(self.root)
        win.title("Скачать модель")
        win.configure(bg="#1e1f24", padx=16, pady=14)
        win.resizable(False, False)
        win.transient(self.root)

        def grab():
            try:
                win.grab_set()
            except tk.TclError:
                pass
        win.after(150, grab)

        tk.Label(win, text="Выбери «мозг» для ИИ.\nСкачивается один раз, "
                 "потом работает без интернета.", bg="#1e1f24", fg="white",
                 font=("Segoe UI", 11), justify=tk.LEFT).pack(anchor="w", pady=(0, 10))

        have = {os.path.basename(m["gguf"]).lower() for m in self.models if m["gguf"]}
        choice = tk.IntVar(value=DEFAULT_CHOICE)
        for i, (title, _repo, fname, desc) in enumerate(CATALOG):
            done = fname.lower() in have
            tk.Radiobutton(win, text=title + ("   ✓ уже есть" if done else ""),
                           variable=choice, value=i, bg="#1e1f24", fg="white",
                           selectcolor="#333", activebackground="#1e1f24",
                           activeforeground="white", font=("Segoe UI", 10, "bold"),
                           state=tk.DISABLED if done else tk.NORMAL).pack(anchor="w")
            tk.Label(win, text="      " + desc, bg="#1e1f24", fg="#999999",
                     font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 6))

        bar = ttk.Progressbar(win, length=420, mode="determinate", maximum=100)
        bar.pack(pady=(8, 4))
        info = tk.Label(win, text="", bg="#1e1f24", fg="#bbbbbb", font=("Segoe UI", 9),
                        wraplength=420, justify=tk.LEFT)
        info.pack()
        btn = tk.Button(win, text="Скачать", width=16, bg="#16a34a", fg="white",
                        relief=tk.FLAT, font=("Segoe UI", 10, "bold"))
        btn.pack(pady=(8, 0))

        def start():
            title, repo, fname, _ = CATALOG[choice.get()]
            if fname.lower() in have:
                return
            btn.config(state=tk.DISABLED, text="Скачиваю...")
            self.downloading = True
            win.protocol("WM_DELETE_WINDOW", lambda: None)
            threading.Thread(target=self.download_model,
                             args=(repo, fname, win, bar, info, btn), daemon=True).start()
        btn.config(command=start)

    def resolve_url(self, repo, fname):
        """Прямая ссылка; если файла с таким именем нет — ищем похожий в репозитории."""
        url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
        try:
            req = urllib.request.Request(url, headers=UA, method="HEAD")
            urllib.request.urlopen(req, timeout=20).close()
            return url, fname
        except urllib.error.HTTPError as e:
            if e.code != 404:
                return url, fname
        quant = fname.split("B-", 1)[-1]          # например "Q4_K_M.gguf"
        req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}", headers=UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            files = [s["rfilename"] for s in json.loads(r.read().decode("utf-8"))["siblings"]]
        for f in files:
            if f.endswith(quant) and "mmproj" not in f.lower():
                return f"https://huggingface.co/{repo}/resolve/main/{f}", os.path.basename(f)
        raise RuntimeError("модель не найдена на сервере")

    def download_model(self, repo, fname, win, bar, info, btn):
        def ui(func, **kw):
            self.root.after(0, lambda: func(**kw))

        def prog(got, total, speed):
            pct = got * 100 / total if total else 0
            left = (total - got) / 1e6 / speed if speed > 0 and total else 0
            ui(bar.config, value=pct)
            ui(info.config, text=f"{got/1e9:.2f} из {total/1e9:.2f} ГБ  •  {speed:.1f} МБ/с  •  "
                                 f"осталось ~{int(left // 60)} мин")
        try:
            url, real = self.resolve_url(repo, fname)
            dest = os.path.join(MODELS_DIR, real)
            download_file(url, dest, prog)
        except Exception as e:
            self.downloading = False
            ui(info.config, text=f"Ошибка: {e}. Проверь интернет и нажми ещё раз — "
                                 "загрузка продолжится с того же места.")
            ui(btn.config, state=tk.NORMAL, text="Продолжить")
            self.root.after(0, lambda: win.protocol("WM_DELETE_WINDOW", win.destroy))
            return
        self.downloading = False

        def finish():
            win.destroy()
            self.write(f"(Модель «{stem_of(real)}» скачана!)\n", "info")
            self.reload_models()
            for i, m in enumerate(self.models):
                if m["gguf"] == dest:
                    self.combo.current(i)
                    self.start_model(m)
        self.root.after(0, finish)

    # ----- запуск модели -----
    def on_model_change(self, _event=None):
        i = self.combo.current()
        if 0 <= i < len(self.models):
            self.start_model(self.models[i])

    def stop_model(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None
            self.base = None

    def start_model(self, model):
        self.busy = True
        self.write(f"\n(Загружаю модель «{model['name']}»...)\n", "info")
        self.set_status("загрузка...")
        threading.Thread(target=self._start_thread, args=(model,), daemon=True).start()

    def _try_launch(self, cmd, port, label):
        """Запускает процесс и ждёт, пока модель будет готова. True — успех."""
        base = f"http://127.0.0.1:{port}"
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, creationflags=NO_WINDOW,
                cwd=os.path.dirname(cmd[0]))
        except OSError as e:
            self.say(f"[Не удалось запустить: {e}]\n", "info")
            self.proc = None
            return False
        for _ in range(600):                       # до 5 минут на загрузку
            if self.proc.poll() is not None:
                self.proc = None
                return False
            if server_ready(base):
                self.base = base
                return True
            time.sleep(0.5)
        self.stop_model()
        return False

    def _start_thread(self, model):
        self.stop_model()

        if self.engine:
            gguf = model["gguf"]
            if not gguf and model["lf"]:
                gguf = os.path.join(MODELS_DIR, model["name"] + ".gguf")
                self.say("(Один раз готовлю эту модель для видеокарты...)\n", "info")
                marks = set()

                def prog(done, total):
                    step = int(done * 100 / total) // 25 * 25
                    if step not in marks:
                        marks.add(step)
                        self.say(f"  {step}%\n", "info")
                try:
                    extract_gguf(model["lf"], gguf, prog)
                    model["gguf"] = gguf
                except Exception as e:
                    self.say(f"[Не получилось: {e}. Запускаю на процессоре.]\n", "info")
                    gguf = None
            if gguf:
                # сначала целиком на видеокарте, потом частично, потом на процессоре
                for ngl, label in (("99", "видеокарта"), ("20", "видеокарта + процессор"),
                                   ("0", "процессор")):
                    port = free_port()
                    cmd = [self.engine, "-m", gguf, "--host", "127.0.0.1",
                           "--port", str(port), "-c", str(CONTEXT), "-ngl", ngl, "--jinja"]
                    if self._try_launch(cmd, port, label):
                        self.on_ready(model, label)
                        return
                    if ngl == "99":
                        self.say("(Целиком в видеопамять не влезла, пробую иначе...)\n", "info")

        if model["lf"]:                              # запасной вариант — llamafile
            for extra in (["--server", "--nobrowser"], ["--server"]):
                port = free_port()
                cmd = [model["lf"]] + extra + ["--host", "127.0.0.1", "--port", str(port)]
                if self._try_launch(cmd, port, "процессор"):
                    self.on_ready(model, "процессор")
                    return

        self.say("[Модель не запустилась. Попробуй модель поменьше или пришли скрин.]\n", "info")
        self.set_status("ошибка")

    def on_ready(self, model, label):
        def done():
            if self.history:
                self.write(f"(Помню {len(self.history)} прошлых сообщений)\n", "info")
            self.write(f"(Работаю на: {label})\n", "info")
            self.write(f"\n{AI_NAME}: ", "ai")
            self.write("Готова! О чём поговорим?\n\n")
            icon = "⚡" if "видеокарта" in label else "●"
            self.status.config(text=f"{icon} {model['name']} — {label}")
            self.busy = False
        self.root.after(0, done)

    # ----- чат -----
    def on_enter(self, event):
        if event.state & 0x1:
            return None
        self.send()
        return "break"

    def clear_memory(self):
        self.history = []
        save_history(self.history)
        self.write("(Память очищена)\n\n", "info")

    def send(self):
        text = self.entry.get("1.0", tk.END).strip()
        if not text or self.busy or not self.base:
            return
        self.entry.delete("1.0", tk.END)
        self.write("Ты: ", "you")
        self.write(text + "\n\n")
        self.history.append({"role": "user", "content": text})
        self.busy = True
        self.write(f"{AI_NAME}: ", "ai")
        threading.Thread(target=self.ask_ai, args=(self.base,), daemon=True).start()

    def ask_ai(self, base):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history[-MEMORY_SIZE:]
        body = json.dumps({
            "model": "local", "messages": messages,
            "stream": True, "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False},   # без долгих «размышлений»
        }).encode("utf-8")
        req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        answer, thinking_shown = "", False
        filt = ThinkFilter()
        md = MarkdownStream()

        def show(text, final=False):
            for seg, tag in (md.flush() if final else md.feed(text)):
                self.say(seg, tag)
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                    if delta.get("reasoning_content") and not thinking_shown and not answer:
                        thinking_shown = True
                        self.say("(думает...) ", "info")
                    piece = filt.feed(delta.get("content") or "")
                    if piece:
                        if not answer:
                            piece = piece.lstrip()
                        answer += piece
                        show(piece)
            tail = filt.flush()
            if tail:
                answer += tail
                show(tail)
            show("", final=True)
        except urllib.error.URLError:
            self.say("\n[Модель не отвечает. Выбери её заново в списке сверху.]\n", "info")
        except Exception as e:
            self.say(f"\n[Ошибка: {e}]\n", "info")

        if answer.strip():
            self.history.append({"role": "assistant", "content": answer.strip()})
            save_history(self.history)
        else:
            self.history.pop()
        self.say("\n\n")
        self.busy = False

    def on_close(self):
        self.stop_model()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    ChatApp(root)
    root.mainloop()
