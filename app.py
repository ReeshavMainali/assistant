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
]

TOOL_FUNCTIONS = {"web_search": web_search, "get_weather": get_weather}


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
                    try:
                        result = func(**args)
                    except Exception as e:
                        result = f"error running {name}: {e}"

                self.messages.append(
                    {"role": "tool", "content": result, "name": name}
                )

        self.write_line("stopped after too many tool call rounds", "system")


if __name__ == "__main__":
    ChatApp().run()