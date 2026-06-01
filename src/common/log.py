"""Structured console logger with tagged, timestamped output for pipeline steps, tools, and state changes."""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from blinker import Signal
from rich.console import Console
from rich.markup import escape
from rich.rule import Rule

_console = Console(highlight=False)


@dataclass(frozen=True)
class LogEvent:
    """A single logged line, broadcast to listeners for out-of-band rendering (e.g. an in-browser overlay)."""

    tag: str
    color: str
    msg: str


#: Emitted once per logged line with a LogEvent as the sender. Import this object and connect a
#: receiver `(event: LogEvent) -> None` to mirror output elsewhere (e.g. the in-page overlay).
#: Blinker holds receivers weakly by default.
log_signal = Signal()


def _ts() -> str:
    """Return current time as HH:MM:SS."""
    return datetime.now().strftime("%H:%M:%S")


def _emit(color: str, tag: str, msg: str) -> None:
    """Print a tagged, timestamped line, dimming any TOOL:/WHY: qualifier so the action name stands out."""
    prefix, sep, name = tag.partition(":")
    tag_markup = (
        f"[dim]{escape(prefix)}{sep}[/][bold {color}]{escape(name)}[/]" if sep else f"[bold {color}]{escape(tag)}[/]"
    )
    _console.print(f"[dim]\\[[/]{tag_markup}[dim]][/]  [dim]{_ts()}[/]  {escape(msg)}")
    log_signal.send(LogEvent(tag=tag, color=color, msg=msg))


def step(tag: str, msg: str) -> None:
    """Log a pipeline step."""
    _emit("cyan", tag, msg)


def tool(name: str, msg: str) -> None:
    """Log a tool invocation."""
    _emit("blue", f"TOOL:{name}", msg)


def reason(name: str, msg: str) -> None:
    """Log the model's stated justification for the action it is about to take."""
    _emit("bright_yellow", f"WHY:{name}", msg)


def state(msg: str) -> None:
    """Log a state mutation."""
    _emit("magenta", "STATE", msg)


def ok(msg: str) -> None:
    """Log a success."""
    _emit("green", "OK", msg)


def warn(msg: str) -> None:
    """Log a warning."""
    _emit("yellow", "WARN", msg)


def err(msg: str) -> None:
    """Log an error."""
    _emit("red", "ERR", msg)


def info(msg: str) -> None:
    """Log an informational message."""
    _emit("bright_black", "INFO", msg)


def block(tag: str, color: str, content: str) -> None:
    """Print multi-line content (prompts, reasoning) with a header rule."""
    _console.print(Rule(f"[{color}]{escape(tag)}[/]", style=color))
    _console.print(escape(content.strip()))
    _console.print(Rule(style=color))
    log_signal.send(LogEvent(tag=tag, color=color, msg=content.strip()))


def _fmt_args(args: str | dict[str, Any] | None) -> str:
    if args is None:
        return ""
    if isinstance(args, str):
        try:
            parsed: dict[str, Any] = json.loads(args)
        except json.JSONDecodeError:
            return args[:200]
    else:
        parsed = args
    parts = []
    for k, v in parsed.items():
        v_str = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else repr(v)
        parts.append(f"{k}={v_str[:80]}")
    return ", ".join(parts)[:300]


def dump_messages(messages: list[Any]) -> None:
    """Print the agent's reasoning trace from a completed run - text, tool calls, and returns."""
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ThinkingPart, ToolCallPart, ToolReturnPart

    _console.print(Rule("[dim]agent trace[/dim]", style="bright_black"))
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ThinkingPart) and part.content.strip():
                    block("THINK", "bright_black", part.content.strip())
                elif isinstance(part, TextPart) and part.content.strip():
                    block("REASON", "white", part.content.strip())
                elif isinstance(part, ToolCallPart):
                    _emit("cyan", "CALL", f"{part.tool_name}({_fmt_args(part.args)})")
        elif isinstance(msg, ModelRequest):
            for req_part in msg.parts:
                if isinstance(req_part, ToolReturnPart):
                    content = str(req_part.content)[:600]
                    _emit("bright_black", "RETN", f"{req_part.tool_name} → {content}")
    _console.print(Rule(style="bright_black"))
