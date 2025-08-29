"""
Pipecat-compatible window control functions.
Provides clean interfaces for LLM tool calls to control application windows.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json
import re
from window_control import WindowController
from action_runner import ActionRunner
from sequence_recorder import SequenceRecorder
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams


# Global controller instance (singleton)
_controller = None
_recorder: Optional[SequenceRecorder] = None
_last_recorded_actions: Optional[List[Dict[str, Any]]]= None

# Sequences index and storage
try:
    from window_control import CACHE_DIR  # reuse existing cache dir
except Exception:
    CACHE_DIR = Path.home() / ".pipecat-dictation"

SEQUENCES_INDEX_FILE = CACHE_DIR / "sequences_index.json"
SEQUENCES_DIR = Path.cwd() / "sequences"

def _slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9\-_. ]+", "", s)
    s = re.sub(r"\s+", "-", s)
    return s or "sequence"

def _load_sequences_index() -> Dict[str, Any]:
    if not SEQUENCES_INDEX_FILE.exists():
        return {"sequences": {}, "updated": datetime.now().isoformat()}
    try:
        with open(SEQUENCES_INDEX_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"sequences": {}, "updated": datetime.now().isoformat()}

def _save_sequences_index(data: Dict[str, Any]) -> None:
    SEQUENCES_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    data["updated"] = datetime.now().isoformat()
    with open(SEQUENCES_INDEX_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _get_controller() -> WindowController:
    """Get or create the global window controller."""
    global _controller
    if _controller is None:
        _controller = WindowController()
    return _controller


# ============================================================================
# Pipecat Tool Functions
# ============================================================================


def list_windows() -> Dict[str, List[Dict[str, str]]]:
    """
    Get the list of all remembered windows.

    Returns:
        Dictionary with 'windows' key containing list of window info
    """
    controller = _get_controller()

    windows = []
    for name, info in controller.window_map.items():
        window_dict = {
            "name": name,
            "title": info.title or "Unknown",
            "class": info.wm_class or "Unknown",
            "is_last_used": name == controller.last_used_window,
        }
        windows.append(window_dict)

    # Sort by last used (most recent first)
    windows.sort(key=lambda x: x["is_last_used"], reverse=True)

    return {
        "success": True,
        "windows": windows,
        "count": len(windows),
        "last_used": controller.last_used_window or "none",
    }


async def handle_list_windows(params: FunctionCallParams):
    """
    Handle list_windows function call for Pipecat.
    """
    result = list_windows()
    await params.result_callback(result)


def remember_window(name: str, wait_seconds: int = 3) -> Dict[str, any]:
    """
    Remember/save the currently focused window with a given name.

    Args:
        name: Name to save this window as
        wait_seconds: Seconds to wait before capturing (default: 3)

    Returns:
        Dictionary with success status and window info
    """
    controller = _get_controller()

    # Sanitize the name
    name = name.strip()
    if not name:
        return {"success": False, "error": "Window name cannot be empty"}

    try:
        # The remember_window method will handle the countdown
        success = controller.remember_window(name, wait_seconds)

        if success and name in controller.window_map:
            info = controller.window_map[name]
            return {
                "success": True,
                "name": name,
                "window": {
                    "title": info.title or "Unknown",
                    "class": info.wm_class or "Unknown",
                    "position": list(info.position) if info.position else None,
                },
                "message": f"Successfully saved window '{name}'",
            }
        else:
            return {"success": False, "error": "Failed to capture window information"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def handle_remember_window(params: FunctionCallParams):
    """
    Handle remember_window function call for Pipecat.
    """
    name = params.arguments.get("name")
    wait_seconds = params.arguments.get("wait_seconds", 3)
    result = remember_window(name, wait_seconds)
    await params.result_callback(result)


def send_text_to_window(
    text: str, window_name: Optional[str] = None, send_newline: bool = True
) -> Dict[str, any]:
    """
    Send text to a specific remembered window.

    Args:
        text: The text to send to the window
        window_name: Name of the window to send to (None for last used)
        send_newline: Whether to send Enter key after the text (default: True)

    Returns:
        Dictionary with success status
    """
    controller = _get_controller()

    # Check if we have any windows
    if not controller.window_map:
        return {"success": False, "error": "No windows remembered. Use remember_window first."}

    # Validate window name if provided
    if window_name and window_name not in controller.window_map:
        return {
            "success": False,
            "error": f"Window '{window_name}' not found",
            "available_windows": list(controller.window_map.keys()),
        }

    try:
        # Special casing "escape" to send escape key
        if text == "escape":
            controller.send_key_to_window("escape", window_name)
            return {
                "success": True,
                "message": "Escape sent to window",
                "window_used": window_name,
            }

        # Send the text
        controller.send_keystrokes_to_window(text, window_name)

        # Send newline if requested
        if send_newline:
            controller.send_key_to_window("enter", window_name)

        # Determine which window was used
        target_window = window_name or controller.last_used_window or "default"

        return {
            "success": True,
            "message": f"Sent text to window '{target_window}'",
            "window_used": target_window,
            "text_length": len(text),
            "newline_sent": send_newline,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def handle_send_text_to_window(params: FunctionCallParams):
    """
    Handle send_text_to_window function call for Pipecat.
    """
    text = params.arguments.get("text")
    window_name = params.arguments.get("window_name", None)
    send_newline = params.arguments.get("send_newline", True)
    result = send_text_to_window(text, window_name, send_newline)
    await params.result_callback(result)


def focus_window(window_name: Optional[str] = None) -> Dict[str, any]:
    """
    Focus a specific window or the last used window.

    Args:
        window_name: Name of window to focus (None for last used)

    Returns:
        Dictionary with success status
    """
    controller = _get_controller()

    if not controller.window_map:
        return {"success": False, "error": "No windows remembered"}

    if window_name and window_name not in controller.window_map:
        return {
            "success": False,
            "error": f"Window '{window_name}' not found",
            "available_windows": list(controller.window_map.keys()),
        }

    try:
        success = controller.focus_window(window_name)
        target = window_name or controller.last_used_window or "default"

        if success:
            return {
                "success": True,
                "message": f"Focused window '{target}'",
                "window_focused": target,
            }
        else:
            return {"success": False, "error": f"Failed to focus window '{target}'"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def handle_focus_window(params: FunctionCallParams):
    """
    Handle focus_window function call for Pipecat.
    """
    window_name = params.arguments.get("window_name", None)
    result = focus_window(window_name)
    await params.result_callback(result)


# ============================================================================
# Pipecat Function Schemas
# ============================================================================


list_windows_schema = FunctionSchema(
    name="list_windows",
    description="Get the list of all remembered windows that can receive text input",
    properties={},
    required=[],
)


remember_window_schema = FunctionSchema(
    name="remember_window",
    description="Remember/save the currently focused window with a custom name for later use",
    properties={
        "name": {
            "type": "string",
            "description": "A memorable name for this window (e.g., 'editor', 'terminal', 'browser')",
        },
        "wait_seconds": {
            "type": "integer",
            "description": "Seconds to wait before capturing the window (default: 3)",
            "default": 3,
            "minimum": 1,
            "maximum": 10,
        },
    },
    required=["name"],
)


send_text_to_window_schema = FunctionSchema(
    name="send_text_to_window",
    description="Send text to a specific remembered window",
    properties={
        "text": {"type": "string", "description": "The text to type into the window"},
        "window_name": {
            "type": "string",
            "description": "Name of the window to send text to (omit to use last focused window)",
        },
        "send_newline": {
            "type": "boolean",
            "description": "Whether to press Enter after sending the text (default: true)",
            "default": True,
        },
    },
    required=["text"],
)


focus_window_schema = FunctionSchema(
    name="focus_window",
    description="Focus/activate a specific remembered window",
    properties={
        "window_name": {
            "type": "string",
            "description": "Name of the window to focus (omit to focus last used window)",
        }
    },
    required=[],
)


# ============================================================================
# Function Registry for Pipecat
# ============================================================================

WINDOW_CONTROL_FUNCTIONS = {
    "list_windows": (list_windows, list_windows_schema),
    "remember_window": (remember_window, remember_window_schema),
    "send_text_to_window": (send_text_to_window, send_text_to_window_schema),
    "focus_window": (focus_window, focus_window_schema),
}

WINDOW_CONTROL_HANDLERS = {
    "list_windows": handle_list_windows,
    "remember_window": handle_remember_window,
    "send_text_to_window": handle_send_text_to_window,
    "focus_window": handle_focus_window,
}


def get_window_control_schemas() -> List[FunctionSchema]:
    """Get all window control function schemas for Pipecat."""
    return [schema for _, schema in WINDOW_CONTROL_FUNCTIONS.values()]


def get_window_control_handlers() -> Dict[str, callable]:
    """Get all window control function handlers for Pipecat."""
    return WINDOW_CONTROL_HANDLERS


# ============================================================================
# Action Sequences (batch execution)
# ============================================================================


def run_actions(actions, variables=None) -> Dict[str, any]:
    """Run a batch of UI actions defined as a list of dicts.

    Args:
        actions: List of action dicts, each with a 'type' field.
        variables: Optional initial variables (e.g., captured points).
    """
    runner = ActionRunner(_get_controller())
    result = runner.run(actions, initial_vars=variables or {})
    return result


async def handle_run_actions(params: FunctionCallParams):
    actions = params.arguments.get("actions")
    variables = params.arguments.get("variables", None)
    result = run_actions(actions, variables)
    await params.result_callback(result)


def run_actions_file(file_path: str, variables=None) -> Dict[str, any]:
    """Load a JSON action list from file and execute it."""
    import json
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        return {"success": False, "error": f"File not found: {file_path}"}
    try:
        with open(path, "r") as f:
            actions = json.load(f)
        if not isinstance(actions, list):
            return {"success": False, "error": "Invalid sequence file (expected a list)"}
    except Exception as e:
        return {"success": False, "error": f"Failed to read file: {e}"}

    return run_actions(actions, variables)


async def handle_run_actions_file(params: FunctionCallParams):
    file_path = params.arguments.get("file_path")
    variables = params.arguments.get("variables", None)
    result = run_actions_file(file_path, variables)
    await params.result_callback(result)


run_actions_schema = FunctionSchema(
    name="run_actions",
    description=(
        "Run a sequence of UI actions like focusing windows, moving/clicking the mouse, "
        "typing text, and waiting. Each action must include a 'type' field."
    ),
    properties={
        "actions": {
            "type": "array",
            "description": "List of actions to perform in order",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Action type: focus_window | move_mouse | hover | click | send_text | key | wait | prompt_point",
                    },
                    "name": {"type": "string", "description": "Window name for focus_window"},
                    "to": {
                        "type": "object",
                        "description": "Target point: {x,y} or {var}",
                    },
                    "button": {"type": "string"},
                    "count": {"type": "integer"},
                    "interval": {"type": "number"},
                    "seconds": {"type": "number"},
                    "text": {"type": "string"},
                    "send_newline": {"type": "boolean"},
                    "window": {"type": "string", "description": "Target window for send_text/key"},
                    "key": {"type": "string"},
                    "var": {"type": "string", "description": "Variable name for prompt_point"},
                    "message": {"type": "string"},
                    "countdown": {"type": "integer"}
                },
                "required": ["type"],
            },
        },
        "variables": {
            "type": "object",
            "description": "Initial variables map, e.g., named points {var: {x,y}}",
        },
    },
    required=["actions"],
)

# Register new schema and handler
WINDOW_CONTROL_FUNCTIONS["run_actions"] = (run_actions, run_actions_schema)
WINDOW_CONTROL_HANDLERS["run_actions"] = handle_run_actions

run_actions_file_schema = FunctionSchema(
    name="run_actions_file",
    description="Run a sequence of UI actions loaded from a JSON file",
    properties={
        "file_path": {
            "type": "string",
            "description": "Path to JSON file containing an array of action objects",
        },
        "variables": {
            "type": "object",
            "description": "Optional initial variables (e.g., named points)",
        },
    },
    required=["file_path"],
)

WINDOW_CONTROL_FUNCTIONS["run_actions_file"] = (
    run_actions_file,
    run_actions_file_schema,
)
WINDOW_CONTROL_HANDLERS["run_actions_file"] = handle_run_actions_file


# ============================================================================
# Named Sequences (save/list/run/delete)
# ============================================================================


def save_sequence(
    name: str,
    actions: Optional[List[Dict[str, Any]]] = None,
    file_path: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Save a sequence under a friendly name and update the index.

    If actions is omitted, uses the last recorded actions if available.
    If file_path is omitted, saves to sequences/<slug>.json.
    """
    global _last_recorded_actions

    name = (name or "").strip()
    if not name:
        return {"success": False, "error": "Name is required"}

    if actions is None:
        if file_path is None and _last_recorded_actions:
            actions = _last_recorded_actions
        elif file_path is None:
            return {"success": False, "error": "No actions provided and no recent recording available"}

    # Determine target file
    path: Path
    if file_path:
        path = Path(file_path)
    else:
        SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)
        slug = _slugify(name)
        path = SEQUENCES_DIR / f"{slug}.json"
        if path.exists() and not overwrite:
            # if exists, try unique suffix
            i = 2
            while (SEQUENCES_DIR / f"{slug}-{i}.json").exists():
                i += 1
            path = SEQUENCES_DIR / f"{slug}-{i}.json"

    # Write actions if provided
    if actions is not None:
        try:
            with open(path, "w") as f:
                json.dump(actions, f, indent=2)
        except Exception as e:
            return {"success": False, "error": f"Failed to write file: {e}"}

    # Update index
    idx = _load_sequences_index()
    seqs = idx.get("sequences", {})
    entry = seqs.get(name, {})
    now = datetime.now().isoformat()
    if not entry:
        entry = {"file": str(path), "created": now, "updated": now}
    else:
        entry.update({"file": str(path), "updated": now})
    seqs[name] = entry
    idx["sequences"] = seqs
    _save_sequences_index(idx)

    # Return metadata
    length = None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            length = len(data)
    except Exception:
        pass

    return {
        "success": True,
        "name": name,
        "file": str(path),
        "count": length,
    }


async def handle_save_sequence(params: FunctionCallParams):
    name = params.arguments.get("name")
    actions = params.arguments.get("actions", None)
    file_path = params.arguments.get("file_path", None)
    overwrite = params.arguments.get("overwrite", False)
    result = save_sequence(name, actions, file_path, overwrite)
    await params.result_callback(result)


def list_sequences() -> Dict[str, Any]:
    idx = _load_sequences_index()
    seqs = idx.get("sequences", {})
    out: List[Dict[str, Any]] = []
    for name, entry in seqs.items():
        file = entry.get("file")
        exists = Path(file).exists() if file else False
        out.append(
            {
                "name": name,
                "file": file,
                "exists": exists,
                "created": entry.get("created"),
                "updated": entry.get("updated"),
            }
        )
    out.sort(key=lambda x: x.get("updated") or "", reverse=True)
    return {"success": True, "count": len(out), "sequences": out}


async def handle_list_sequences(params: FunctionCallParams):
    result = list_sequences()
    await params.result_callback(result)


def delete_sequence(name: str, remove_file: bool = True) -> Dict[str, Any]:
    idx = _load_sequences_index()
    seqs = idx.get("sequences", {})
    if name not in seqs:
        return {"success": False, "error": f"No sequence named '{name}'"}

    entry = seqs.pop(name)
    idx["sequences"] = seqs
    _save_sequences_index(idx)

    removed_file = None
    if remove_file and entry.get("file"):
        path = Path(entry["file"])
        try:
            if path.exists():
                path.unlink()
            removed_file = str(path)
        except Exception as e:
            return {"success": False, "error": f"Failed to delete file: {e}"}

    return {"success": True, "name": name, "removed_file": removed_file}


async def handle_delete_sequence(params: FunctionCallParams):
    name = params.arguments.get("name")
    remove_file = params.arguments.get("remove_file", True)
    result = delete_sequence(name, remove_file)
    await params.result_callback(result)


def run_sequence(name: str, variables=None) -> Dict[str, Any]:
    idx = _load_sequences_index()
    entry = idx.get("sequences", {}).get(name)
    if not entry:
        return {"success": False, "error": f"No sequence named '{name}'"}
    file_path = entry.get("file")
    if not file_path:
        return {"success": False, "error": f"Sequence '{name}' has no file path"}
    return run_actions_file(file_path, variables)


async def handle_run_sequence(params: FunctionCallParams):
    name = params.arguments.get("name")
    variables = params.arguments.get("variables", None)
    result = run_sequence(name, variables)
    await params.result_callback(result)


save_sequence_schema = FunctionSchema(
    name="save_sequence",
    description=(
        "Save a UI action sequence under a friendly name and update the sequences index. "
        "If actions are omitted, uses the most recent recorded actions."
    ),
    properties={
        "name": {"type": "string", "description": "Sequence name"},
        "actions": {
            "type": "array",
            "description": "List of action objects to save (optional)",
            "items": {"type": "object"},
        },
        "file_path": {
            "type": "string",
            "description": "Explicit file path for the sequence JSON (optional)",
        },
        "overwrite": {
            "type": "boolean",
            "description": "Overwrite if target file exists (default: false)",
            "default": False,
        },
    },
    required=["name"],
)


list_sequences_schema = FunctionSchema(
    name="list_sequences",
    description="List saved sequences from the index with file paths and timestamps",
    properties={},
    required=[],
)


delete_sequence_schema = FunctionSchema(
    name="delete_sequence",
    description="Delete a saved sequence by name (and optionally its file)",
    properties={
        "name": {"type": "string", "description": "Sequence name to delete"},
        "remove_file": {
            "type": "boolean",
            "description": "Also delete the underlying JSON file (default: true)",
            "default": True,
        },
    },
    required=["name"],
)


run_sequence_schema = FunctionSchema(
    name="run_sequence",
    description="Run a named sequence from the index",
    properties={
        "name": {"type": "string", "description": "Sequence name to run"},
        "variables": {
            "type": "object",
            "description": "Optional initial variables for the sequence",
        },
    },
    required=["name"],
)


# Register named sequence tools
WINDOW_CONTROL_FUNCTIONS["save_sequence"] = (save_sequence, save_sequence_schema)
WINDOW_CONTROL_HANDLERS["save_sequence"] = handle_save_sequence

WINDOW_CONTROL_FUNCTIONS["list_sequences"] = (list_sequences, list_sequences_schema)
WINDOW_CONTROL_HANDLERS["list_sequences"] = handle_list_sequences

WINDOW_CONTROL_FUNCTIONS["delete_sequence"] = (delete_sequence, delete_sequence_schema)
WINDOW_CONTROL_HANDLERS["delete_sequence"] = handle_delete_sequence

WINDOW_CONTROL_FUNCTIONS["run_sequence"] = (run_sequence, run_sequence_schema)
WINDOW_CONTROL_HANDLERS["run_sequence"] = handle_run_sequence


# ============================================================================
# Action Recording (start/stop)
# ============================================================================


def start_action_recording(
    default_window: Optional[str] = None,
    focus_hotkeys: Optional[List[Dict[str, Any]]] = None,
    min_wait: float = 0.25,
    double_click_window: float = 0.35,
) -> Dict[str, Any]:
    """Start recording clicks/keys in the background.

    Args:
        default_window: Window name to attach to send_text/key actions.
        focus_hotkeys: List like [{"fn": 1, "window": "bot"}, ...] to insert focus_window via F-keys.
        min_wait: Insert wait when idle gap exceeds this many seconds.
        double_click_window: Merge clicks closer than this many seconds.
    """
    global _recorder
    if _recorder and _recorder.is_running():
        return {"success": False, "error": "Recording already in progress"}

    # Convert hotkeys list to map of {fn_number: window_name}
    key_map: Dict[int, str] = {}
    if focus_hotkeys:
        for item in focus_hotkeys:
            try:
                fn = int(item.get("fn"))
                name = str(item.get("window"))
                if 1 <= fn <= 12 and name:
                    key_map[fn] = name
            except Exception:
                pass

    _recorder = SequenceRecorder(
        default_window=default_window,
        window_key_map=key_map,
        min_wait=min_wait,
        double_click_window=double_click_window,
    )
    _recorder.start()
    return {
        "success": True,
        "message": "Recording started. Call stop_action_recording to finish.",
        "default_window": default_window,
        "focus_hotkeys": key_map,
    }


async def handle_start_action_recording(params: FunctionCallParams):
    default_window = params.arguments.get("default_window", None)
    focus_hotkeys = params.arguments.get("focus_hotkeys", None)
    min_wait = params.arguments.get("min_wait", 0.25)
    double_click_window = params.arguments.get("double_click_window", 0.35)
    result = start_action_recording(default_window, focus_hotkeys, min_wait, double_click_window)
    await params.result_callback(result)


def stop_action_recording(
    save_to_file: Optional[str] = None,
    append: bool = False,
) -> Dict[str, Any]:
    """Stop background recording and return recorded actions. Optionally save to file.

    Args:
        save_to_file: Path to write JSON array of actions.
        append: If true and file exists, append to existing action list.
    """
    global _recorder
    if not _recorder or not _recorder.is_running():
        return {"success": False, "error": "No active recording"}

    actions = _recorder.stop()
    # store as last recorded for convenience
    global _last_recorded_actions
    _last_recorded_actions = actions
    _recorder = None

    file_path = None
    if save_to_file:
        from pathlib import Path
        import json

        path = Path(save_to_file)
        try:
            if append and path.exists():
                with open(path, "r") as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
                to_write = existing + actions
            else:
                to_write = actions
            with open(path, "w") as f:
                json.dump(to_write, f, indent=2)
            file_path = str(path)
        except Exception as e:
            return {"success": False, "error": f"Failed to save file: {e}", "actions": actions}

    return {
        "success": True,
        "count": len(actions),
        "actions": actions,
        "file": file_path,
    }


async def handle_stop_action_recording(params: FunctionCallParams):
    save_to_file = params.arguments.get("save_to_file", None)
    append = params.arguments.get("append", False)
    result = stop_action_recording(save_to_file, append)
    await params.result_callback(result)


start_action_recording_schema = FunctionSchema(
    name="start_action_recording",
    description=(
        "Start recording mouse clicks and keystrokes into a JSON action sequence. "
        "Use stop_action_recording to finish and retrieve actions."
    ),
    properties={
        "default_window": {
            "type": "string",
            "description": "Optional window name to attach to send_text/key actions",
        },
        "focus_hotkeys": {
            "type": "array",
            "description": "List of F-key mappings to focus windows during recording",
            "items": {
                "type": "object",
                "properties": {
                    "fn": {
                        "type": "integer",
                        "description": "Function key number (1..12) for F1..F12",
                        "minimum": 1,
                        "maximum": 12,
                    },
                    "window": {
                        "type": "string",
                        "description": "Window name to focus when this F-key is pressed",
                    },
                },
                "required": ["fn", "window"],
            },
        },
        "min_wait": {
            "type": "number",
            "description": "Insert wait action when idle gap exceeds this many seconds",
            "default": 0.25,
        },
        "double_click_window": {
            "type": "number",
            "description": "Merge rapid clicks within this time window (seconds)",
            "default": 0.35,
        },
    },
    required=[],
)


stop_action_recording_schema = FunctionSchema(
    name="stop_action_recording",
    description=(
        "Stop the background action recording and return the captured actions. "
        "Optionally save them to a file (append or overwrite)."
    ),
    properties={
        "save_to_file": {
            "type": "string",
            "description": "Optional path to save the action list as JSON",
        },
        "append": {
            "type": "boolean",
            "description": "Append to existing file if it exists (default: false)",
            "default": False,
        },
    },
    required=[],
)


# Register new schemas and handlers
WINDOW_CONTROL_FUNCTIONS["start_action_recording"] = (
    start_action_recording,
    start_action_recording_schema,
)
WINDOW_CONTROL_HANDLERS["start_action_recording"] = handle_start_action_recording

WINDOW_CONTROL_FUNCTIONS["stop_action_recording"] = (
    stop_action_recording,
    stop_action_recording_schema,
)
WINDOW_CONTROL_HANDLERS["stop_action_recording"] = handle_stop_action_recording
