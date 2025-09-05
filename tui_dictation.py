import asyncio
from typing import Optional, Any
import argparse
import os
import sys

from textual.app import ComposeResult
from textual.widgets import Header, Footer, Static
from textual.containers import Vertical, Horizontal

from loguru import logger

from voice_tui_kit.core.base_app import BotTUIBase
from voice_tui_kit.widgets.text_list_panel import TextListPanel
from voice_tui_kit.core.utils.imports import import_bot_module


class DictationTUI(BotTUIBase):
    """Special-purpose TUI for dictation and sending.

    Layout:
    - Status bar
    - Main split (1fr): left Dictated, right Sent
    - Messages list (6 rows)
    Overlays:
    - Ctrl+N RTVI view, Ctrl+L Log view (provided by base)
    - Ctrl+M mute/unmute toggle (UI state only)
    """

    CSS = """
    Screen { layout: vertical; }
    #status { height: 3; border: round $accent; padding: 0 1; }
    #main { layout: vertical; height: 1fr; }
    #split { layout: horizontal; height: 1fr; }
    #dictated, #sent { height: 1fr; border: round $primary; }
    #messages { height: 6; border: solid $primary; }
    #messages ListItem { padding: 0 1; }
    /* RTVI overlay sizing */
    #rtvi_panes { layout: horizontal; height: 1fr; }
    """

    # Note: ctrl+m is treated as Enter in many terminals; use F2 for mute
    BINDINGS = BotTUIBase.BINDINGS + [
        ("f2", "toggle_mute", "Mute/Unmute"),
    ]

    def __init__(self, bot_module) -> None:  # type: ignore[no-untyped-def]
        super().__init__(bot_module)
        self.dictated: Optional[TextListPanel] = None
        self.sent: Optional[TextListPanel] = None
        self.messages: Optional[TextListPanel] = None
        self._last_messages_append_type: Optional[str] = None
        self._muted: bool = False
        self._countdown_task: Optional[asyncio.Task] = None
        self._last_connected: bool = False
        self._rtvi_sent_message_id: int = 0

    def compose(self) -> ComposeResult:  # type: ignore[override]
        yield Header(show_clock=True)
        with Vertical(id="main"):
            self.status = Static("Status: initializing", id="status")
            yield self.status
            with Horizontal(id="split"):
                self.dictated = TextListPanel(id="dictated")
                yield self.dictated
                self.sent = TextListPanel(id="sent")
                yield self.sent
            self.messages = TextListPanel(id="messages")
            yield self.messages
            # Overlays expected by base (RTVI + Syslog), initially hidden
            from voice_tui_kit.widgets.rtvi_list_panel import RTVIListPanel
            from voice_tui_kit.widgets.syslog_panel import SyslogPanel

            with Horizontal(id="rtvi_panes"):
                self.rtvi_inbox = RTVIListPanel(id="inbox", classes="log")
                yield self.rtvi_inbox
                self.rtvi_outbox = RTVIListPanel(id="outbox", classes="log")
                yield self.rtvi_outbox
            self.syslog = SyslogPanel(id="syslog", classes="log")
            self.syslog.display = False
            yield self.syslog
        yield Footer()

    async def on_mount(self) -> None:  # type: ignore[override]
        await super().on_mount()
        # Add in-list headers for clarity
        try:
            from textual.widgets import ListItem as _LI

            if self.dictated is not None:
                await self.dictated.append(_LI(Static("Dictated:")))
                self.dictated.hide_placeholder()
            if self.sent is not None:
                await self.sent.append(_LI(Static("Sent:")))
                self.sent.hide_placeholder()
            if self.messages is not None:
                await self.messages.append(_LI(Static("Messages:")))
                self.messages.hide_placeholder()
        except Exception:
            pass

    async def _on_status(self, connected: bool) -> None:  # type: ignore[override]
        await super()._on_status(connected)
        self._last_connected = connected
        self._update_status_suffix()

    def _update_status_suffix(self) -> None:
        if self.status is None:
            return
        base = "Status: connected" if self._last_connected else "Status: disconnected"
        if self._muted:
            base += " — Muted"
        self.status.update(base)

    async def _on_inbound(self, payload: Any) -> None:  # type: ignore[override]
        await super()._on_inbound(payload)
        await self._handle_server_message_events(payload)
        await self._maybe_append_message(payload)

    async def _on_outbound(self, payload: Any) -> None:  # type: ignore[override]
        await super()._on_outbound(payload)
        await self._handle_server_message_events(payload)
        await self._maybe_append_message(payload)

    async def _handle_server_message_events(self, payload: Any) -> None:
        try:
            if not isinstance(payload, dict) or not payload.get("type") == "server-message":
                return
            logger.debug(f"_handle_server_message_events: {payload}")
            type = payload.get("data", {}).get("type")
            data = payload.get("data", {}).get("data", {})
            if type == "sent-text":
                edited_text = data.get("edited_text")
                raw_text = data.get("raw_text")
                window_name = data.get("window_name")
                if raw_text:
                    await self.dictated.append_text(raw_text)
                if edited_text:
                    await self.sent.append_text(f"|{window_name}| {edited_text}")
            elif type == "remember-window":
                name = data.get("name") or "window"
                seconds = int(data.get("seconds") or 3)
                self._start_countdown(name, seconds)

        except Exception:
            pass

    def _start_countdown(self, name: str, seconds: int) -> None:
        if self._countdown_task and not self._countdown_task.done():
            self._countdown_task.cancel()

        async def _run():
            try:
                for remaining in range(seconds, -1, -1):
                    if self.status:
                        suffix = " — Muted" if self._muted else ""
                        self.status.update(f"Choosing window {name} — {remaining}s{suffix}")
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                return
            finally:
                # revert to transport status
                self._update_status_suffix()

        self._countdown_task = asyncio.create_task(_run())

    async def _maybe_append_message(self, payload: Any) -> None:
        try:
            if not isinstance(payload, dict) or self.messages is None:
                return
            t = payload.get("type")
            data = payload.get("data", {}) if isinstance(payload.get("data"), dict) else {}
            text = data.get("text")
            final = data.get("final")
            if not isinstance(text, str):
                return
            if t == "user-transcription" and final:
                if self._last_messages_append_type == "User":
                    await self.messages.append_text_to_last_item(" " + text)
                else:
                    await self.messages.append_text_item("User: " + text)
                self._last_messages_append_type = "User"
            elif t == "bot-transcription":
                if self._last_messages_append_type == "Bot":
                    await self.messages.append_text_to_last_item(" " + text)
                else:
                    await self.messages.append_text_item("Bot: " + text)
                self._last_messages_append_type = "Bot"
        except Exception:
            pass

    async def action_toggle_mute(self) -> None:
        self._muted = not self._muted
        self._update_status_suffix()
        # Send control payload to the bot to toggle mute/unmute
        try:
            payload = {
                "id": f"mute-{self._rtvi_sent_message_id}",
                "label": "rtvi-ai",
                "type": "client-message",
                "data": {"t": "mute-unmute", "d": {"mute": bool(self._muted)}},
            }
            await self.transport_mgr.send_app_message(payload)
            self._rtvi_sent_message_id += 1
            if self.syslog is not None:
                self.syslog.write_line(
                    f"[info] {'Muted' if self._muted else 'Unmuted'} (sent client-message)"
                )
        except Exception as e:
            if self.syslog is not None:
                self.syslog.write_line(f"[error] Failed to send mute toggle: {e}")


def main(argv: Optional[list[str]] = None) -> int:
    # Optional: route Textual logs to a separate file
    os.environ.setdefault("TEXTUAL_LOG", "tui_textual.log")
    os.environ.setdefault("TEXTUAL", "debug")

    # Lightweight logging consistent with tui_demo
    try:
        logger.remove()
    except Exception:
        pass
    logger.add(sys.stderr, level="INFO")
    logger.add("tui_dictation.app.log", level="DEBUG", rotation="1 MB", retention=5)

    parser = argparse.ArgumentParser(description="Dictation/Sent TUI app")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--bot", help="Python module path, e.g. my_pkg.my_bot")
    g.add_argument("--file", help="Path to a bot Python file, e.g. ./bot.py")
    parser.add_argument("--inline", action="store_true", help="Run Textual inline (debug)")
    args = parser.parse_args(argv)

    mod = import_bot_module(args.bot or args.file)
    app = DictationTUI(mod)
    if args.inline:
        app.run(inline=True, inline_no_clear=True)
    else:
        app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
