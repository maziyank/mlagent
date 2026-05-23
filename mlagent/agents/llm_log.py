"""Terminal logging for deep-agent / LLM invocations."""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def _message_count(state: dict[str, Any]) -> int:
    messages = state.get("messages") or []
    return len(messages) if isinstance(messages, list) else 0


def _summarize_messages(state: dict[str, Any], tail: int = 3) -> str:
    messages = state.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return "(no messages)"
    lines: list[str] = []
    for msg in messages[-tail:]:
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        if role is None and isinstance(msg, dict):
            role = msg.get("role") or msg.get("type", "?")
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content", "")
        if isinstance(content, list):
            content = str(content)[:200]
        elif isinstance(content, str):
            content = content[:200].replace("\n", " ")
        else:
            content = str(content)[:200]
        tool_calls = getattr(msg, "tool_calls", None) or (
            msg.get("tool_calls") if isinstance(msg, dict) else None
        )
        extra = f" tools={len(tool_calls)}" if tool_calls else ""
        lines.append(f"{role}: {content}{extra}")
    return "\n".join(lines)


def _extract_tool_names(update: dict[str, Any]) -> list[str]:
    names: list[str] = []
    messages = update.get("messages")
    if not isinstance(messages, list):
        return names
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None) or (
            msg.get("tool_calls") if isinstance(msg, dict) else None
        )
        if not tool_calls:
            continue
        for tc in tool_calls:
            name = getattr(tc, "name", None) or (
                tc.get("name") if isinstance(tc, dict) else None
            )
            if name:
                names.append(name)
    return names


def _log_stream_part_v2(part: dict[str, Any], *, depth: int = 0) -> None:
    prefix = "  " * depth
    part_type = part.get("type", "?")
    ns = part.get("ns") or ()
    ns_label = ".".join(ns) if ns else "root"
    data = part.get("data")

    if part_type == "updates" and isinstance(data, dict):
        for node, update in data.items():
            if node.startswith("__"):
                console.print(f"{prefix}[dim]state {node}[/dim]")
                continue
            tools = _extract_tool_names(update) if isinstance(update, dict) else []
            tool_hint = f" → tools: {', '.join(tools)}" if tools else ""
            msg_n = _message_count(update) if isinstance(update, dict) else 0
            console.print(
                f"{prefix}[blue]node[/blue] {ns_label}/{node} "
                f"[dim](messages={msg_n})[/dim]{tool_hint}"
            )
    elif part_type == "messages" and isinstance(data, tuple) and len(data) >= 1:
        msg, meta = data[0], data[1] if len(data) > 1 else {}
        node = (meta or {}).get("langgraph_node", ns_label)
        chunk = getattr(msg, "content", "") or ""
        if isinstance(chunk, str) and chunk.strip():
            preview = chunk.replace("\n", " ")[:80]
            console.print(f"{prefix}[magenta]token[/magenta] [{node}] {preview}")
    elif part_type == "tasks" and isinstance(data, dict):
        status = data.get("status") or data.get("event") or "task"
        name = data.get("name") or data.get("id") or ns_label
        console.print(f"{prefix}[yellow]task[/yellow] {name}: {status}")
    elif part_type == "values" and isinstance(data, dict):
        console.print(
            f"{prefix}[dim]values[/dim] messages={_message_count(data)} "
            f"keys={list(data.keys())[:8]}"
        )


def _log_stream_chunk_v1(chunk: Any, mode: str | None = None) -> None:
    if mode == "updates" and isinstance(chunk, dict):
        for node, update in chunk.items():
            if node.startswith("__"):
                continue
            tools = _extract_tool_names(update) if isinstance(update, dict) else []
            tool_hint = f" → {', '.join(tools)}" if tools else ""
            console.print(f"  [blue]node[/blue] {node}{tool_hint}")
    elif mode == "messages" and isinstance(chunk, tuple) and chunk:
        msg = chunk[0]
        text = getattr(msg, "content", "") or ""
        if isinstance(text, str) and text.strip():
            console.print(f"  [magenta]token[/magenta] {text[:80]}")


def invoke_agent_with_logging(
    agent: Any,
    input_state: dict[str, Any],
    *,
    stage: str,
    purpose: str,
    model: str,
    run_id: str,
    enabled: bool = True,
    run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke a deep agent graph and log LLM / graph state to the terminal."""
    invoke_kwargs: dict[str, Any] = {}
    if run_config:
        invoke_kwargs["config"] = run_config

    if not enabled:
        return agent.invoke(input_state, **invoke_kwargs)

    prompt_chars = 0
    messages = input_state.get("messages") or []
    if messages:
        last = messages[-1]
        content = getattr(last, "content", None) or (
            last.get("content") if isinstance(last, dict) else ""
        )
        prompt_chars = len(str(content))

    header = Text()
    header.append("LLM call ", style="bold magenta")
    header.append(f"stage={stage} ", style="cyan")
    header.append(f"purpose={purpose} ", style="yellow")
    header.append(f"model={model} ", style="green")
    header.append(f"run={run_id}", style="dim")
    console.print(Panel(header, title="Agent invoke", border_style="magenta"))
    console.print(
        f"[dim]input:[/dim] messages={len(messages)} prompt_chars≈{prompt_chars}"
    )

    start = time.perf_counter()
    final_state: dict[str, Any] | None = None
    update_count = 0
    token_events = 0

    try:
        stream = agent.stream(
            input_state,
            stream_mode=["updates", "messages", "tasks", "values"],
            version="v2",
            **invoke_kwargs,
        )
        for part in stream:
            if isinstance(part, dict) and "type" in part:
                _log_stream_part_v2(part)
                if part.get("type") == "values" and isinstance(part.get("data"), dict):
                    final_state = part["data"]
                if part.get("type") == "updates":
                    update_count += 1
                if part.get("type") == "messages":
                    token_events += 1
            elif isinstance(part, tuple) and len(part) == 2:
                mode, data = part
                _log_stream_chunk_v1(data, str(mode))
                if mode == "updates":
                    update_count += 1
                if mode == "messages":
                    token_events += 1
    except TypeError:
        # Older graphs without version= parameter
        try:
            for mode, data in agent.stream(
                input_state, stream_mode=["updates", "messages"], **invoke_kwargs
            ):
                _log_stream_chunk_v1(data, str(mode))
                if mode == "updates":
                    update_count += 1
                if mode == "values" and isinstance(data, dict):
                    final_state = data
        except Exception:
            console.print("[dim]stream unavailable, using invoke()[/dim]")
            final_state = agent.invoke(input_state, **invoke_kwargs)
    except Exception as exc:
        console.print(f"[red]stream error:[/red] {exc} — falling back to invoke()")
        final_state = agent.invoke(input_state, **invoke_kwargs)

    if final_state is None:
        final_state = agent.invoke(input_state, **invoke_kwargs)

    elapsed = time.perf_counter() - start
    console.print(
        f"[green]LLM done[/green] stage={stage} "
        f"elapsed={elapsed:.1f}s updates={update_count} tokens={token_events} "
        f"messages={_message_count(final_state)}"
    )
    summary = _summarize_messages(final_state)
    if summary != "(no messages)":
        console.print(Panel(summary, title="Final messages (tail)", border_style="dim"))

    todos = final_state.get("todos")
    if todos:
        console.print(f"[dim]todos:[/dim] {todos}")

    return final_state
