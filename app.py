import asyncio
from pathlib import Path

import ollama
import requests
from ddgs import DDGS
from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Header, Footer, Input, Select, Static, Markdown

MODELS = {
    "gemma4:e4b": "gemma4:e4b",
    "qwen3.5:4b": "qwen3.5:4b",
}
DEFAULT_MODEL = "gemma4:e4b"

MAX_TOOL_ROUNDS = 3

# Sandbox root: the tools can never read or write outside this "current directory".
BASE_DIR = Path(__file__).resolve().parent
# Default place files are stored.
WORKSPACE_DIR = BASE_DIR / "workspace"
# Active storage directory for writes; selectable via the set_directory tool or /dir.
STORAGE_DIR = WORKSPACE_DIR

WEATHER_CODES = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    80: "rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


def web_search(query: str, max_results: int = 5) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    if not results:
        return "No results found."
    return "\n".join(
        f"- {r['title']}: {r['body']} ({r['href']})" for r in results
    )


def get_weather(location: str) -> str:
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1},
        timeout=10,
    ).json()
    if not geo.get("results"):
        return f"couldn't find a location matching '{location}'"

    place = geo["results"][0]
    lat, lon = place["latitude"], place["longitude"]
    label = ", ".join(
        p for p in [place.get("name"), place.get("admin1"), place.get("country")] if p
    )

    forecast = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code",
            "timezone": "auto",
        },
        timeout=10,
    ).json()

    current = forecast.get("current")
    if not current:
        return f"weather lookup for {label} failed"

    condition = WEATHER_CODES.get(current["weather_code"], "unknown conditions")
    return (
        f"{label}: {condition}, {current['temperature_2m']}°C "
        f"(feels like {current['apparent_temperature']}°C), "
        f"humidity {current['relative_humidity_2m']}%, "
        f"wind {current['wind_speed_10m']} km/h"
    )


def _check_inside(target: Path, original: str) -> Path:
    if not target.is_relative_to(BASE_DIR):
        raise ValueError(f"path '{original}' is outside the current directory")
    return target


def _resolve_read_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return _check_inside(p.resolve(), path)
    # Relative paths are looked up in the storage directory first,
    # then in the current directory (project root).
    primary = (STORAGE_DIR / p).resolve()
    if primary.exists() and primary.is_relative_to(BASE_DIR):
        return primary
    return _check_inside((BASE_DIR / p).resolve(), path)


def _resolve_write_path(path: str) -> Path:
    p = Path(path)
    target = p.resolve() if p.is_absolute() else (STORAGE_DIR / p).resolve()
    return _check_inside(target, path)


def set_directory(path: str) -> str:
    global STORAGE_DIR
    p = Path(path)
    target = p.resolve() if p.is_absolute() else (BASE_DIR / p).resolve()
    _check_inside(target, path)
    target.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR = target
    return f"storage directory set to {STORAGE_DIR}"


def read_file(path: str) -> str:
    target = _resolve_read_path(path)
    if not target.exists():
        return f"error: no file found at '{path}'"
    if target.is_dir():
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        return "that is a directory; contents:\n" + "\n".join(entries)
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"error: '{path}' is not a text file"


def write_file(path: str, content: str) -> str:
    target = _resolve_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} characters to '{target}'"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. Use this for anything "
                "that needs up-to-date facts, news, or details you're not sure of."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "How many results to return (default 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather conditions for a place.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, optionally with country, e.g. 'Hetauda, Nepal'",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a text file. Relative paths are looked up in the "
                "storage directory first, then in the current directory "
                "(project root). If given a directory, it lists its "
                "contents instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace, e.g. 'notes/todo.txt'",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write (create or overwrite) a text file. Paths are "
                "relative to the current storage directory (the workspace "
                "by default) and cannot go outside the current directory. "
                "Missing parent folders are created automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the storage directory, e.g. 'notes/todo.txt'",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_directory",
            "description": (
                "Change the storage directory used by write_file (and "
                "looked up first by read_file). Paths are relative to the "
                "current directory (project root) and cannot go outside "
                "it. Use '.' for the current directory itself. The default "
                "is 'workspace'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the current directory, e.g. 'docs' or '.'",
                    },
                },
                "required": ["path"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "web_search": web_search,
    "get_weather": get_weather,
    "read_file": read_file,
    "write_file": write_file,
    "set_directory": set_directory,
}


class ChatApp(App):
    CSS = """
    #chat {
        height: 1fr;
        padding: 1 2;
    }
    .user {
        color: cyan;
        margin-bottom: 1;
    }
    .assistant-header {
        color: white;
        text-style: bold;
    }
    .assistant-md {
        margin-bottom: 1;
        padding: 0;
    }
    .system {
        color: grey;
        margin-bottom: 1;
    }
    #model-select {
        dock: top;
        width: 30;
    }
    """

    BINDINGS = [
        ("ctrl+l", "clear_chat", "Clear"),
        ("ctrl+t", "toggle_think", "Toggle think"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.model = DEFAULT_MODEL
        self.think = False
        self.messages: list[dict] = []
        self.client = ollama.AsyncClient()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Select(
            options=[(name, name) for name in MODELS],
            value=DEFAULT_MODEL,
            id="model-select",
        )
        yield VerticalScroll(id="chat")
        yield Input(placeholder="Type a message and press enter...", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#prompt", Input).focus()
        WORKSPACE_DIR.mkdir(exist_ok=True)
        self.write_line(f"storage directory: {STORAGE_DIR} (root: {BASE_DIR})", "system")
        self.fetch_models()

    @work(exclusive=True)
    async def fetch_models(self) -> None:
        try:
            res = await self.client.list()
            model_names = []
            if hasattr(res, "models"):
                for m in res.models:
                    name = getattr(m, "model", None)
                    if name:
                        model_names.append(name)
            elif isinstance(res, dict):
                for m in res.get("models", []):
                    name = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
                    if name:
                        model_names.append(name)

            if model_names:
                select = self.query_one("#model-select", Select)
                select.set_options([(n, n) for n in model_names])
                if self.model not in model_names:
                    self.model = model_names[0]
                    select.value = self.model
        except Exception:
            pass

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "model-select":
            self.model = event.value
            self.write_line(f"switched to {self.model}", "system")

    def write_line(self, text: str, css_class: str) -> Static:
        widget = Static(text, classes=css_class)
        self.query_one("#chat", VerticalScroll).mount(widget)
        widget.scroll_visible()
        return widget

    def write_markdown(self, text: str = "") -> Markdown:
        widget = Markdown(text, classes="assistant-md")
        self.query_one("#chat", VerticalScroll).mount(widget)
        widget.scroll_visible()
        return widget

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return

        if text == "/clear":
            self.action_clear_chat()
            return
        if text.startswith("/save"):
            parts = text.split(maxsplit=1)
            filename = parts[1] if len(parts) > 1 else "chat_history.json"
            self.action_save_chat(filename)
            return
        if text.startswith("/load"):
            parts = text.split(maxsplit=1)
            filename = parts[1] if len(parts) > 1 else "chat_history.json"
            self.action_load_chat(filename)
            return
        if text == "/dir" or text.startswith("/dir "):
            parts = text.split(maxsplit=1)
            if len(parts) > 1:
                try:
                    message = set_directory(parts[1])
                except Exception as e:
                    message = f"error: {e}"
            else:
                message = f"storage directory: {STORAGE_DIR} (root: {BASE_DIR})"
            self.write_line(message, "system")
            return

        self.write_line(f"you> {text}", "user")
        self.messages.append({"role": "user", "content": text})
        self.stream_reply()

    def action_toggle_think(self) -> None:
        self.think = not self.think
        self.write_line(f"thinking mode: {'on' if self.think else 'off'}", "system")

    def action_clear_chat(self) -> None:
        self.messages = []
        self.query_one("#chat", VerticalScroll).remove_children()
        self.write_line("context cleared", "system")

    def action_save_chat(self, filename: str = "chat_history.json") -> None:
        import json
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
            self.write_line(f"chat saved to {filename}", "system")
        except Exception as e:
            self.write_line(f"error saving chat: {e}", "system")

    def action_load_chat(self, filename: str = "chat_history.json") -> None:
        import json
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.messages = data
                scroll = self.query_one("#chat", VerticalScroll)
                scroll.remove_children()
                for msg in self.messages:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if role == "user":
                        self.write_line(f"you> {content}", "user")
                    elif role == "assistant":
                        if content:
                            self.write_line(f"{self.model}>", "assistant-header")
                            self.write_markdown(content)
                        if msg.get("tool_calls"):
                            for call in msg["tool_calls"]:
                                name = call["function"]["name"]
                                args = call["function"]["arguments"]
                                self.write_line(f"→ calling {name}({args})", "system")
                    elif role == "tool":
                        self.write_line(f"[tool result]: {content}", "system")
                self.write_line(f"chat loaded from {filename}", "system")
            else:
                self.write_line("error: invalid chat file format", "system")
        except Exception as e:
            self.write_line(f"error loading chat: {e}", "system")

    @work(exclusive=True)
    async def stream_reply(self) -> None:
        model = self.model

        for _ in range(MAX_TOOL_ROUNDS):
            self.write_line(f"{model}>", "assistant-header")
            reply_widget = self.write_markdown()
            accumulated = ""
            tool_calls = None

            try:
                stream = await self.client.chat(
                    model=model,
                    messages=self.messages,
                    tools=TOOLS,
                    stream=True,
                    think=self.think,
                )
                async for chunk in stream:
                    message = chunk["message"]
                    if message.get("content"):
                        accumulated += message["content"]
                        await reply_widget.update(accumulated)
                        reply_widget.scroll_visible()
                    if message.get("tool_calls"):
                        tool_calls = message["tool_calls"]
            except Exception as e:
                await reply_widget.update(f"*[error] {e}*")
                return

            if not tool_calls:
                if not accumulated:
                    reply_widget.remove()
                self.messages.append({"role": "assistant", "content": accumulated})
                return

            if not accumulated:
                reply_widget.remove()
            self.messages.append(
                {"role": "assistant", "content": accumulated, "tool_calls": tool_calls}
            )

            for call in tool_calls:
                name = call["function"]["name"]
                args = call["function"]["arguments"]
                self.write_line(f"→ calling {name}({args})", "system")

                func = TOOL_FUNCTIONS.get(name)
                if func is None:
                    result = f"error: no such tool '{name}'"
                else:
                    if not isinstance(args, dict):
                        args = {}
                    
                    validation_error = None
                    if name == "web_search":
                        if not args.get("query"):
                            validation_error = "error: 'query' argument is required for web_search"
                    elif name == "get_weather":
                        if not args.get("location"):
                            validation_error = "error: 'location' argument is required for get_weather"
                    elif name == "read_file":
                        if not args.get("path"):
                            validation_error = "error: 'path' argument is required for read_file"
                    elif name == "write_file":
                        if not args.get("path"):
                            validation_error = "error: 'path' argument is required for write_file"
                        elif args.get("content") is None:
                            validation_error = "error: 'content' argument is required for write_file"
                    elif name == "set_directory":
                        if not args.get("path"):
                            validation_error = "error: 'path' argument is required for set_directory"

                    if validation_error:
                        result = validation_error
                    else:
                        try:
                            result = await asyncio.to_thread(func, **args)
                        except Exception as e:
                            result = f"error running {name}: {e}"

                self.messages.append(
                    {"role": "tool", "content": result, "name": name}
                )

        self.write_line("stopped after too many tool call rounds", "system")


if __name__ == "__main__":
    ChatApp().run()